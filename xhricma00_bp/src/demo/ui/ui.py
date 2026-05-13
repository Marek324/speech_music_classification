# demo/ui/ui.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Reflex front-end for the streaming demo.

Runs on localhost; the server captures mic audio via sounddevice and streams
predictions back to the browser through Reflex state.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import wave
from pathlib import Path

import numpy as np
import psutil
import reflex as rx

from src.demo.runner import RUNNERS
from src.demo.source import (
    FileSource,
    MicSource,
    MixedSource,
    ParecordSource,
    list_input_devices,
    list_pulse_monitor_sources,
)
from src.demo.weights_check import (
    ALL_MODELS,
    CLASSIC_MODELS,
    NN_MODELS,
    all_status,
)
from src.nn.variants import VARIANTS

log = logging.getLogger(__name__)

_CLASSIC_LABELS = {
    "decision_tree": "Decision Tree",
    "gmm": "GMM",
    "svm": "SVM",
}
_MODEL_LABELS: dict[str, str] = {
    **_CLASSIC_LABELS,
    "tcn": "TCN (paper)",
    **{name: v.ui_label for name, v in VARIANTS.items()},
}

_LABEL_NAME = {-1: "speech", 0: "background", 1: "music"}
_LABEL_COLOR = {-1: "blue", 0: "gray", 1: "crimson"}

_N_BUCKETS = 200
_UI_HZ = 12.0

_PLAYBACK_SEC = 3.0
_PLAYBACK_FILE = "playback.wav"
_RESOURCE_POLL_SEC = 1.0
_NCPU = psutil.cpu_count(logical=True) or 1

_SOURCE_MIC = "mic"
_SOURCE_MIX = "mic_desktop"
_SOURCE_DESKTOP = "desktop"


class DemoState(rx.State):
    model_name: str = "tcn"
    mode: str = "mic"
    running: bool = False
    predictions: list[dict] = []
    weights_status: dict[str, bool] = {}
    uploaded_file_name: str = ""
    uploaded_file_path: str = ""
    status_msg: str = ""

    mic_device: int = -1
    audio_source_mode: str = _SOURCE_MIC
    desktop_source: str = ""
    desktop_gain_db: int = 0
    devices: list[dict] = []
    monitor_sources: list[dict] = []

    _audio_buffer: bytearray = bytearray()
    _audio_buffer_sr: int = 0
    playback_filename: str = ""

    cpu_percent: float = 0.0
    proc_mem_mb: float = 0.0
    sys_mem_used_gb: float = 0.0
    sys_mem_total_gb: float = 0.0
    sys_mem_percent: float = 0.0
    monitoring: bool = False

    @rx.var
    def mic_options(self) -> list[dict]:
        opts: list[dict] = [{"value": "-1", "label": "System default"}]
        for d in self.devices:
            opts.append({"value": str(d["index"]), "label": d.get("label", d["name"])})
        return opts

    @rx.var
    def desktop_options(self) -> list[dict]:
        return [
            {"value": s["name"], "label": s["description"]}
            for s in self.monitor_sources
        ]

    @rx.var
    def show_mic_picker(self) -> bool:
        return self.audio_source_mode != _SOURCE_DESKTOP

    @rx.var
    def show_desktop_picker(self) -> bool:
        return self.audio_source_mode != _SOURCE_MIC

    @rx.var
    def no_monitors_hint(self) -> str:
        if self.audio_source_mode == _SOURCE_MIC:
            return ""
        if self.monitor_sources:
            return ""
        return (
            "No monitor sources found. Make sure pipewire-pulse (or pulseaudio-utils) "
            "is installed so `pactl` and `parecord` are available, then refresh."
        )

    @rx.var
    def selected_mic_value(self) -> str:
        return str(self.mic_device)

    @rx.var
    def selected_desktop_value(self) -> str:
        return self.desktop_source

    @rx.var
    def current_label_name(self) -> str:
        if not self.predictions:
            return "—"
        return _LABEL_NAME.get(self.predictions[-1]["label"], "—")

    @rx.var
    def current_label_color(self) -> str:
        if not self.predictions:
            return "gray"
        return _LABEL_COLOR.get(self.predictions[-1]["label"], "gray")

    @rx.var
    def has_predictions(self) -> bool:
        return len(self.predictions) > 0

    @rx.var
    def cpu_label(self) -> str:
        return f"{self.cpu_percent:.0f}%"

    @rx.var
    def proc_mem_label(self) -> str:
        return f"{self.proc_mem_mb:.0f} MB"

    @rx.var
    def sys_mem_label(self) -> str:
        return f"{self.sys_mem_used_gb:.1f} / {self.sys_mem_total_gb:.1f} GB"

    @rx.var
    def sys_mem_percent_label(self) -> str:
        return f"{self.sys_mem_percent:.0f}%"

    @rx.event
    def on_load(self):
        """Initialise weights status, audio device lists, and the default selected model."""
        self.weights_status = all_status()
        try:
            self.devices = list_input_devices()
        except Exception as exc:
            log.exception("device enumeration failed")
            self.devices = []
            self.status_msg = f"Device enumeration failed: {exc}"
        try:
            self.monitor_sources = list_pulse_monitor_sources()
        except Exception as exc:
            log.exception("pulse monitor enumeration failed")
            self.monitor_sources = []
            self.status_msg = f"Pulse source enumeration failed: {exc}"
        self.mic_device = -1
        self.audio_source_mode = _SOURCE_MIC
        self.desktop_source = self.monitor_sources[0]["name"] if self.monitor_sources else ""
        preferred = ["tcn", *VARIANTS.keys(), *CLASSIC_MODELS]
        for name in preferred:
            if self.weights_status.get(name):
                self.model_name = name
                return

    @rx.event(background=True)
    async def monitor_resources(self):
        """Poll process + system resource usage at ~1Hz for the lifetime of the session.

        cpu_percent() needs a priming call before it returns meaningful samples,
        and the `monitoring` guard prevents duplicate loops if on_load fires
        more than once (e.g. tab refresh in dev).
        """
        async with self:
            if self.monitoring:
                return
            self.monitoring = True
        proc = psutil.Process()
        proc.cpu_percent(None)
        try:
            while True:
                await asyncio.sleep(_RESOURCE_POLL_SEC)
                try:
                    cpu = proc.cpu_percent(None) / _NCPU
                    rss_mb = proc.memory_info().rss / (1024 ** 2)
                    vm = psutil.virtual_memory()
                except Exception:
                    log.exception("resource sample failed")
                    continue
                async with self:
                    self.cpu_percent = cpu
                    self.proc_mem_mb = rss_mb
                    self.sys_mem_used_gb = (vm.total - vm.available) / (1024 ** 3)
                    self.sys_mem_total_gb = vm.total / (1024 ** 3)
                    self.sys_mem_percent = vm.percent
        finally:
            async with self:
                self.monitoring = False

    @rx.event
    def select_mic_device(self, value: str):
        try:
            self.mic_device = int(value)
        except (TypeError, ValueError):
            self.mic_device = -1

    @rx.event
    def select_desktop_source(self, value: str):
        self.desktop_source = value

    @rx.event
    def set_desktop_gain_db(self, value: list[int | float]):
        v = value[0] if isinstance(value, list) else value
        try:
            self.desktop_gain_db = max(0, min(40, int(v)))
        except (TypeError, ValueError):
            self.desktop_gain_db = 0

    @rx.var
    def desktop_gain_label(self) -> str:
        return f"+{self.desktop_gain_db} dB"

    @rx.event
    def select_audio_source_mode(self, value: str | list[str]):
        v = value[0] if isinstance(value, list) else value
        if v not in (_SOURCE_MIC, _SOURCE_MIX, _SOURCE_DESKTOP):
            return
        self.audio_source_mode = v
        if v != _SOURCE_MIC and not self.desktop_source and self.monitor_sources:
            self.desktop_source = self.monitor_sources[0]["name"]

    @rx.event
    def select_model(self, name: str):
        """Select a model if its weights are present; otherwise post an error message."""
        if not self.weights_status.get(name, False):
            self.status_msg = f"{_MODEL_LABELS.get(name, name)} has no weights on disk."
            return
        self.model_name = name
        self.status_msg = ""
        self.predictions = []
        self._reset_audio_buffer()

    @rx.event
    def set_mode(self, mode: str):
        self.mode = mode
        self.predictions = []
        self._reset_audio_buffer()

    def _reset_audio_buffer(self) -> None:
        self._audio_buffer = bytearray()
        self._audio_buffer_sr = 0
        self.playback_filename = ""

    def _append_audio(self, chunk: np.ndarray, sr: int) -> None:
        """Append a chunk to the rolling playback buffer and trim to ``_PLAYBACK_SEC`` of audio."""
        self._audio_buffer_sr = sr
        keep_bytes = int(sr * _PLAYBACK_SEC) * 4
        self._audio_buffer.extend(chunk.astype(np.float32, copy=False).tobytes())
        excess = len(self._audio_buffer) - keep_bytes
        if excess > 0:
            del self._audio_buffer[:excess]

    @rx.event
    def play_last_3s(self):
        """Write the rolling playback buffer to a WAV file and surface it via the audio element."""
        if not self._audio_buffer or self._audio_buffer_sr <= 0:
            self.status_msg = "No audio captured yet."
            return
        samples = np.frombuffer(bytes(self._audio_buffer), dtype=np.float32)
        int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        upload_dir = Path(rx.get_upload_dir())
        upload_dir.mkdir(parents=True, exist_ok=True)
        out_path = upload_dir / _PLAYBACK_FILE
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._audio_buffer_sr)
            w.writeframes(int16.tobytes())
        self.playback_filename = f"{_PLAYBACK_FILE}?v={int(time.time() * 1000)}"

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Persist the first uploaded file under the upload directory for later file-mode playback."""
        if not files:
            return
        f = files[0]
        data = await f.read()
        upload_dir = Path(rx.get_upload_dir())
        upload_dir.mkdir(parents=True, exist_ok=True)
        fname = f.name or "upload.wav"
        dest = upload_dir / fname
        dest.write_bytes(data)
        self.uploaded_file_name = fname
        self.uploaded_file_path = str(dest)
        self.status_msg = f"Loaded {fname}."

    @rx.event
    def stop(self):
        self.running = False

    @rx.event(background=True)
    async def run_mic(self):
        """Open the configured audio source and stream predictions to the chart until stopped."""
        async with self:
            if self.running:
                return
            if not self.weights_status.get(self.model_name, False):
                self.status_msg = "Selected model has no weights."
                return
            self.running = True
            self.predictions = []
            self._reset_audio_buffer()
            self.status_msg = f"Loading {_MODEL_LABELS[self.model_name]} weights…"
            model_name = self.model_name
            primary = None if self.mic_device < 0 else self.mic_device
            mode = self.audio_source_mode
            desktop_name = self.desktop_source or None
            desktop_gain = float(10 ** (self.desktop_gain_db / 20.0))
            if mode != _SOURCE_MIC and not desktop_name:
                self.running = False
                self.status_msg = "Pick a monitor source for desktop audio."
                return

        try:
            runner = RUNNERS[model_name]()
        except Exception as exc:
            log.exception("runner build failed")
            async with self:
                self.running = False
                self.status_msg = f"Failed to load model: {exc}"
            return

        async with self:
            if not self.running:
                runner.close()
                self.status_msg = "Stopped."
                return
            self.status_msg = {
                _SOURCE_MIC: "Opening input device…",
                _SOURCE_MIX: "Opening mic + desktop…",
                _SOURCE_DESKTOP: "Opening desktop audio…",
            }[mode]

        if mode == _SOURCE_MIC:
            source = MicSource(
                sr=runner.sr, chunk_samples=runner.chunk_samples, device=primary
            )
        elif mode == _SOURCE_DESKTOP:
            source = ParecordSource(
                sr=runner.sr,
                chunk_samples=runner.chunk_samples,
                source_name=desktop_name,
                gain=desktop_gain,
            )
        else:
            source = MixedSource(
                sr=runner.sr,
                chunk_samples=runner.chunk_samples,
                primary_device=primary,
                include_secondary=True,
                secondary_pulse_source=desktop_name,
                secondary_gain=desktop_gain,
            )
        saw_chunk = False
        errored = False
        runner_sr = int(runner.sr)
        bucket_dt = runner.chunk_samples / runner_sr
        flush_every = max(1, round((runner_sr / runner.chunk_samples) / _UI_HZ))
        labels_buf: list[int] = []
        pending_labels: list[int] = []
        try:
            async for chunk in source.chunks():
                preds = runner.push(chunk)
                if preds:
                    pending_labels.extend(p.label for p in preds)
                flush = len(pending_labels) >= flush_every
                async with self:
                    if not self.running:
                        break
                    if not saw_chunk:
                        self.status_msg = "Listening…"
                        saw_chunk = True
                    self._append_audio(chunk, runner_sr)
                    if flush:
                        labels_buf.extend(pending_labels)
                        pending_labels.clear()
                        if len(labels_buf) > _N_BUCKETS:
                            del labels_buf[: len(labels_buf) - _N_BUCKETS]
                        self.predictions = [
                            {"t": round(i * bucket_dt, 3), "label": labels_buf[i]}
                            for i in range(len(labels_buf))
                        ]
        except Exception as exc:
            log.exception("mic loop crashed")
            errored = True
            async with self:
                self.status_msg = f"Mic error: {exc}"
        finally:
            source.close()
            runner.close()
            del source, runner
            gc.collect()
            async with self:
                self.running = False
                if not errored:
                    self.status_msg = "Stopped."

    @rx.event(background=True)
    async def run_file(self):
        """Stream predictions from the previously uploaded file to the chart until stopped."""
        async with self:
            if self.running:
                return
            if not self.uploaded_file_path:
                self.status_msg = "Upload a file first."
                return
            if not self.weights_status.get(self.model_name, False):
                self.status_msg = "Selected model has no weights."
                return
            self.running = True
            self.predictions = []
            self._reset_audio_buffer()
            self.status_msg = f"Loading {_MODEL_LABELS[self.model_name]} weights…"
            model_name = self.model_name
            path = self.uploaded_file_path
            fname = self.uploaded_file_name

        try:
            runner = RUNNERS[model_name]()
        except Exception as exc:
            log.exception("runner build failed")
            async with self:
                self.running = False
                self.status_msg = f"Failed to load model: {exc}"
            return

        async with self:
            if not self.running:
                runner.close()
                self.status_msg = "Stopped."
                return
            self.status_msg = f"Playing {fname}…"

        source = FileSource(
            path=path, sr=runner.sr, chunk_samples=runner.chunk_samples, realtime=True
        )
        errored = False
        runner_sr = int(runner.sr)
        bucket_dt = runner.chunk_samples / runner_sr
        flush_every = max(1, round((runner_sr / runner.chunk_samples) / _UI_HZ))
        labels_buf: list[int] = []
        pending_labels: list[int] = []
        try:
            async for chunk in source.chunks():
                preds = runner.push(chunk)
                if preds:
                    pending_labels.extend(p.label for p in preds)
                flush = len(pending_labels) >= flush_every
                async with self:
                    if not self.running:
                        break
                    self._append_audio(chunk, runner_sr)
                    if flush:
                        labels_buf.extend(pending_labels)
                        pending_labels.clear()
                        if len(labels_buf) > _N_BUCKETS:
                            del labels_buf[: len(labels_buf) - _N_BUCKETS]
                        self.predictions = [
                            {"t": round(i * bucket_dt, 3), "label": labels_buf[i]}
                            for i in range(len(labels_buf))
                        ]
        except Exception as exc:
            log.exception("file loop crashed")
            errored = True
            async with self:
                self.status_msg = f"File error: {exc}"
        finally:
            source.close()
            runner.close()
            del source, runner
            gc.collect()
            async with self:
                self.running = False
                if not errored:
                    self.status_msg = "Done."




def _picker_button(name: str) -> rx.Component:
    has = DemoState.weights_status[name]
    is_selected = DemoState.model_name == name
    return rx.button(
        rx.hstack(
            rx.text(_MODEL_LABELS[name], weight="medium"),
            rx.spacer(),
            rx.cond(
                has,
                rx.icon(tag="check", size=16),
                rx.icon(tag="x", size=16),
            ),
            align="center",
            width="100%",
        ),
        on_click=DemoState.select_model(name),
        variant=rx.cond(is_selected, "solid", "soft"),
        color_scheme=rx.cond(has, "indigo", "gray"),
        disabled=~has,
        size="3",
        width="100%",
    )


def _picker() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Model", size="4"),
            rx.text("Classic", size="1", color_scheme="gray", weight="medium"),
            rx.vstack(
                *[_picker_button(m) for m in CLASSIC_MODELS],
                spacing="2",
                width="100%",
            ),
            rx.box(height="0.5em"),
            rx.text("Neural", size="1", color_scheme="gray", weight="medium"),
            rx.vstack(
                *[_picker_button(m) for m in NN_MODELS],
                spacing="2",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="280px",
        flex_shrink="0",
    )


def _mic_panel() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Classify live audio from the microphone, desktop playback, or both mixed.",
            size="2",
            color_scheme="gray",
        ),
        rx.vstack(
            rx.text("Source", size="1", color_scheme="gray", weight="medium"),
            rx.segmented_control.root(
                rx.segmented_control.item("Mic", value=_SOURCE_MIC),
                rx.segmented_control.item("Mic + Desktop", value=_SOURCE_MIX),
                rx.segmented_control.item("Desktop", value=_SOURCE_DESKTOP),
                value=DemoState.audio_source_mode,
                on_change=DemoState.select_audio_source_mode,
                disabled=DemoState.running,
                width="100%",
            ),
            spacing="1",
            width="100%",
        ),
        rx.cond(
            DemoState.show_mic_picker,
            rx.vstack(
                rx.text("Microphone", size="1", color_scheme="gray", weight="medium"),
                rx.select.root(
                    rx.select.trigger(placeholder="System default", width="100%"),
                    rx.select.content(
                        rx.foreach(
                            DemoState.mic_options,
                            lambda o: rx.select.item(o["label"], value=o["value"]),
                        ),
                    ),
                    value=DemoState.selected_mic_value,
                    on_change=DemoState.select_mic_device,
                    disabled=DemoState.running,
                ),
                spacing="1",
                width="100%",
            ),
            rx.fragment(),
        ),
        rx.cond(
            DemoState.show_desktop_picker,
            rx.vstack(
                rx.text("Desktop source", size="1", color_scheme="gray", weight="medium"),
                rx.select.root(
                    rx.select.trigger(placeholder="Select monitor source", width="100%"),
                    rx.select.content(
                        rx.foreach(
                            DemoState.desktop_options,
                            lambda o: rx.select.item(o["label"], value=o["value"]),
                        ),
                    ),
                    value=DemoState.selected_desktop_value,
                    on_change=DemoState.select_desktop_source,
                    disabled=DemoState.running,
                ),
                rx.cond(
                    DemoState.no_monitors_hint != "",
                    rx.box(
                        rx.text(DemoState.no_monitors_hint, size="1"),
                        padding="0.5em 0.75em",
                        border_radius="6px",
                        background_color="var(--amber-a3)",
                        border="1px solid var(--amber-a6)",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                rx.hstack(
                    rx.text("Desktop gain", size="1", color_scheme="gray", weight="medium"),
                    rx.spacer(),
                    rx.text(DemoState.desktop_gain_label, size="1", color_scheme="gray"),
                    align="center",
                    width="100%",
                ),
                rx.slider(
                    default_value=0,
                    value=[DemoState.desktop_gain_db],
                    on_change=DemoState.set_desktop_gain_db,
                    min=0,
                    max=40,
                    step=1,
                    disabled=DemoState.running,
                    width="100%",
                ),
                rx.text(
                    "Boost the monitor signal — handy when an app's per-stream volume is low.",
                    size="1",
                    color_scheme="gray",
                ),
                spacing="1",
                width="100%",
            ),
            rx.fragment(),
        ),
        rx.hstack(
            rx.button(
                rx.icon(tag="mic", size=16), "Start",
                on_click=DemoState.run_mic,
                disabled=DemoState.running,
                color_scheme="green",
                size="3",
            ),
            rx.button(
                rx.icon(tag="square", size=16), "Stop",
                on_click=DemoState.stop,
                disabled=~DemoState.running,
                color_scheme="red",
                variant="soft",
                size="3",
            ),
            spacing="2",
        ),
        spacing="3",
        width="100%",
        padding_top="0.75em",
    )


def _file_panel() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Upload an audio file and play it through the model.",
            size="2",
            color_scheme="gray",
        ),
        rx.upload.root(
            rx.hstack(
                rx.icon(tag="upload", size=16),
                rx.text(
                    rx.cond(
                        DemoState.uploaded_file_name != "",
                        DemoState.uploaded_file_name,
                        "Drop a file or click to browse",
                    ),
                    size="2",
                ),
                spacing="2",
                align="center",
            ),
            id="audio_upload",
            multiple=False,
            accept={"audio/*": [".wav", ".flac", ".ogg", ".mp3"]},
            on_drop=DemoState.handle_upload(rx.upload_files(upload_id="audio_upload")),
            border="1px dashed var(--gray-a6)",
            padding="1em",
            border_radius="8px",
            width="100%",
            cursor="pointer",
        ),
        rx.hstack(
            rx.button(
                rx.icon(tag="play", size=16), "Play",
                on_click=DemoState.run_file,
                disabled=DemoState.running | (DemoState.uploaded_file_path == ""),
                color_scheme="green",
                size="3",
            ),
            rx.button(
                rx.icon(tag="square", size=16), "Stop",
                on_click=DemoState.stop,
                disabled=~DemoState.running,
                color_scheme="red",
                variant="soft",
                size="3",
            ),
            spacing="2",
        ),
        spacing="3",
        width="100%",
        padding_top="0.75em",
    )


def _resource_strip() -> rx.Component:
    def stat(icon: str, label: str, value: rx.Var | str, sub: rx.Var | str | None = None) -> rx.Component:
        children = [
            rx.icon(tag=icon, size=14, color="var(--gray-9)"),
            rx.text(label, size="1", color_scheme="gray"),
            rx.text(value, size="2", weight="medium"),
        ]
        if sub is not None:
            children.append(rx.text(sub, size="1", color_scheme="gray"))
        return rx.hstack(*children, spacing="2", align="center")

    return rx.hstack(
        stat("cpu", "CPU", DemoState.cpu_label),
        rx.box(width="1px", height="16px", background_color="var(--gray-a5)"),
        stat("memory-stick", "Process", DemoState.proc_mem_label),
        rx.box(width="1px", height="16px", background_color="var(--gray-a5)"),
        stat("server", "System RAM", DemoState.sys_mem_label, DemoState.sys_mem_percent_label),
        spacing="3",
        align="center",
        padding="0.4em 0.75em",
        border="1px solid var(--gray-a5)",
        border_radius="6px",
        background_color="var(--gray-a2)",
    )


def _status_row() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.text("Current:", size="2", color_scheme="gray"),
            rx.badge(
                DemoState.current_label_name,
                color_scheme=DemoState.current_label_color,
                variant="solid",
                size="2",
            ),
            align="center",
            spacing="2",
        ),
        rx.spacer(),
        rx.cond(
            DemoState.status_msg != "",
            rx.hstack(
                rx.cond(
                    DemoState.running,
                    rx.spinner(size="1"),
                    rx.fragment(),
                ),
                rx.text(DemoState.status_msg, size="2", color_scheme="gray"),
                align="center",
                spacing="2",
            ),
            rx.fragment(),
        ),
        align="center",
        width="100%",
    )


def _chart() -> rx.Component:
    empty = rx.center(
        rx.vstack(
            rx.icon(tag="activity", size=28, color="var(--gray-8)"),
            rx.text(
                "Waiting for audio…",
                size="2",
                color_scheme="gray",
            ),
            align="center",
            spacing="2",
        ),
        height="320px",
        width="100%",
        border="1px dashed var(--gray-a5)",
        border_radius="8px",
    )
    chart = rx.recharts.line_chart(
        rx.recharts.line(
            data_key="label",
            stroke="var(--indigo-9)",
            stroke_width=2,
            type_="stepAfter",
            dot=False,
            is_animation_active=False,
        ),
        rx.recharts.x_axis(
            data_key="t",
            type_="number",
            domain=["dataMin", "dataMax"],
            tick_formatter="function(v){return v.toFixed(1)+'s'}",
        ),
        rx.recharts.y_axis(
            domain=[-1.2, 1.2],
            ticks=[-1, 0, 1],
            tick_formatter="function(v){return v===-1?'speech':v===1?'music':v===0?'background':''}",
            width=80,
        ),
        rx.recharts.cartesian_grid(stroke_dasharray="3 3", vertical=False),
        data=DemoState.predictions,
        width="100%",
        height=320,
    )
    return rx.cond(DemoState.has_predictions, chart, empty)


def _playback_row() -> rx.Component:
    return rx.hstack(
        rx.button(
            rx.icon(tag="rotate-ccw", size=16),
            "Play last 3s",
            on_click=DemoState.play_last_3s,
            color_scheme="indigo",
            variant="soft",
            size="2",
        ),
        rx.cond(
            DemoState.playback_filename != "",
            rx.el.audio(
                src=rx.get_upload_url(DemoState.playback_filename),
                controls=True,
                auto_play=True,
                key=DemoState.playback_filename,
                style={"height": "32px", "flex": "1"},
            ),
            rx.fragment(),
        ),
        spacing="3",
        align="center",
        width="100%",
    )


def _right_panel() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        rx.hstack(rx.icon(tag="mic", size=14), rx.text("Mic"), spacing="1", align="center"),
                        value="mic",
                    ),
                    rx.tabs.trigger(
                        rx.hstack(rx.icon(tag="file-audio", size=14), rx.text("File"), spacing="1", align="center"),
                        value="file",
                    ),
                ),
                rx.tabs.content(_mic_panel(), value="mic"),
                rx.tabs.content(_file_panel(), value="file"),
                value=DemoState.mode,
                on_change=DemoState.set_mode,
            ),
            rx.separator(),
            _status_row(),
            _chart(),
            _playback_row(),
            spacing="4",
            width="100%",
        ),
        flex_grow="1",
    )


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading("Speech / Music Streaming Demo", size="7"),
                    rx.text(
                        "Pick a model, choose mic or file, watch live predictions.",
                        size="3",
                        color_scheme="gray",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                _resource_strip(),
                align="center",
                width="100%",
            ),
            rx.hstack(
                _picker(),
                _right_panel(),
                spacing="4",
                align="start",
                width="100%",
            ),
            spacing="5",
            width="100%",
        ),
        padding_y="2em",
        padding_x="1.5em",
        max_width="1100px",
    )


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="indigo",
        radius="medium",
        scaling="100%",
    ),
)
app.add_page(index, on_load=[DemoState.on_load, DemoState.monitor_resources])
