# src/demo/source.py
# Marek Hric

"""Server-side audio producers for the Reflex demo.

Both ``MicSource`` and ``FileSource`` expose ``async def chunks()`` yielding
mono float32 numpy arrays of ``chunk_samples`` length at the requested sample
rate. ``MicSource`` uses sounddevice (PortAudio resamples as needed);
``FileSource`` uses soundfile + librosa.resample.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

log = logging.getLogger(__name__)


# ALSA plug-chain aliases that wrap a real device — drop them so the picker
# isn't cluttered with sysdefault/dmix/iec958/etc. Anything not matching one
# of these prefixes is kept (including JACK-host names like "Trust GXT…").
_ALSA_ALIAS_PREFIXES = (
    "sysdefault", "samplerate", "speexrate", "surround", "front:",
    "iec958", "hdmi", "null", "upmix", "vdownmix", "usbstream",
    "speex", "oss", "dmix", "dsnoop",
)


def _keep_input_device(name: str) -> bool:
    low = name.lower().strip()
    if low == "jack":
        return False
    return not any(low.startswith(p) for p in _ALSA_ALIAS_PREFIXES)


@contextlib.contextmanager
def _silence_fd(fd: int):
    """Temporarily redirect a low-level file descriptor to /dev/null.

    PortAudio/ALSA print directly to stderr from C when an ``Pa_OpenStream`` call
    fails (e.g. ``paInvalidSampleRate`` for hw: devices), bypassing Python's
    logging. The MicSource fallback already recovers from these failures, so we
    silence the C-level fd around the first open attempt.
    """
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


def list_input_devices() -> list[dict]:
    """Return the useful sounddevice input devices as
    ``[{index, name, label, hostapi, channels, is_monitor}]``.

    ``label`` is the user-facing string ("name (hostapi)") so duplicates from
    different host APIs (e.g. the same mic on ALSA and JACK) are
    distinguishable in the picker.

    ``is_monitor`` is a best-effort heuristic — ``True`` if the device name
    contains "monitor" (case-insensitive), indicating a PulseAudio/PipeWire
    loopback-of-sink source. On setups where the PortAudio backend only
    exposes a generic ``pulse``/``pipewire`` device, no monitors will show
    up and the UI surfaces a hint telling the user to set a monitor as the
    default source (``pavucontrol`` / ``wpctl set-default``).
    """
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    out: list[dict] = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d.get("name", f"device {i}")
        if not _keep_input_device(name):
            continue
        host = hostapis[d["hostapi"]]["name"]
        out.append(
            {
                "index": i,
                "name": name,
                "label": f"{name} ({host})",
                "hostapi": host,
                "channels": int(d["max_input_channels"]),
                "is_monitor": "monitor" in name.lower(),
            }
        )
    return out


def list_pulse_monitor_sources() -> list[dict]:
    """Enumerate Pulse/Pipewire ``*.monitor`` sources via ``pactl``.

    Returns ``[{name, description}]`` for each monitor source. Uses pactl's
    JSON output so works on both PulseAudio and pipewire-pulse. Returns an
    empty list if pactl isn't installed or fails — the UI surfaces a hint
    in that case.
    """
    if shutil.which("pactl") is None:
        return []
    try:
        out = subprocess.run(
            ["pactl", "-f", "json", "list", "sources"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        log.exception("pactl list sources failed")
        return []
    try:
        sources = json.loads(out.stdout)
    except json.JSONDecodeError:
        log.exception("pactl returned invalid JSON")
        return []
    monitors: list[dict] = []
    for s in sources:
        name = s.get("name", "")
        if name.endswith(".monitor"):
            monitors.append(
                {"name": name, "description": s.get("description", name)}
            )
    return monitors


class ParecordSource:
    """Capture from a Pulse/Pipewire source by name via a ``parecord`` subprocess.

    Yields fixed-size float32 mono chunks at ``sr``. Bypasses PortAudio
    entirely so monitor sources are addressable regardless of the system
    default source — works wherever pipewire-pulse or PulseAudio is running.
    ``source_name=None`` lets parecord pick the Pulse default source.
    """

    def __init__(
        self,
        sr: int,
        chunk_samples: int,
        source_name: Optional[str] = None,
        gain: float = 1.0,
    ) -> None:
        self.sr = sr
        self.chunk_samples = chunk_samples
        self.source_name = source_name
        # Linear multiplier applied before yielding. Useful for monitor sources
        # whose per-app sink-input volume sits below 100% — see ui.py.
        self.gain = float(gain)
        self._proc: Optional[asyncio.subprocess.Process] = None

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Yield fixed-size float32 mono chunks captured from the configured Pulse source."""
        # --latency-msec keeps parecord's internal buffer small so stdout flushes
        # promptly. Without it, PulseAudio defaults to a buffer (~1-2s of audio)
        # that delays the first chunk by seconds and makes the mixer block.
        cmd = [
            "parecord",
            f"--rate={self.sr}",
            "--channels=1",
            "--format=float32le",
            "--raw",
            "--latency-msec=50",
        ]
        if self.source_name:
            cmd.append(f"--device={self.source_name}")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "parecord not found — install pulseaudio-utils (or pipewire-pulse)"
            ) from exc

        bytes_per_chunk = self.chunk_samples * 4  # float32 = 4 bytes
        try:
            while True:
                try:
                    data = await self._proc.stdout.readexactly(bytes_per_chunk)
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        arr = np.frombuffer(exc.partial, dtype="<f4")
                        if arr.size:
                            tail = arr.astype(np.float32, copy=True)
                            if self.gain != 1.0:
                                tail = np.clip(tail * self.gain, -1.0, 1.0)
                            pad = np.zeros(self.chunk_samples - tail.size, dtype=np.float32)
                            yield np.concatenate([tail, pad])
                    if self._proc.returncode not in (None, 0):
                        err = (await self._proc.stderr.read()).decode(errors="replace")
                        raise RuntimeError(
                            f"parecord exited with code {self._proc.returncode}: {err.strip()}"
                        )
                    return
                arr = np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
                if self.gain != 1.0:
                    arr = np.clip(arr * self.gain, -1.0, 1.0)
                yield arr
        finally:
            self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                self._proc.terminate()
        except ProcessLookupError:
            pass
        self._proc = None


class MicSource:
    """Open a PortAudio input stream and yield fixed-size chunks at ``sr``.

    Raw ALSA hardware devices (``hw:…``) only accept their native sample
    rate. If opening at the requested ``sr`` fails, we fall back to the
    device's default rate and resample each block on the way out.
    """

    def __init__(
        self,
        sr: int,
        chunk_samples: int,
        device: Optional[int] = None,
    ) -> None:
        self.sr = sr
        self.chunk_samples = chunk_samples
        self.device = device
        self._queue: Optional[asyncio.Queue[np.ndarray]] = None
        self._stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("mic status: %s", status)
        if self._queue is None or self._loop is None:
            return
        frame = indata[:, 0].copy()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def _open_stream(self, sd, dev_sr: int, blocksize: int):
        return sd.InputStream(
            samplerate=dev_sr,
            channels=1,
            blocksize=blocksize,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Yield fixed-size float32 mono chunks from the PortAudio input stream."""
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

        # Try the model's sample rate first; if the device refuses it, fall
        # back to its native rate and resample. PortAudio/ALSA print to stderr
        # from C on the failed attempt, so silence fd 2 around it.
        dev_sr = self.sr
        dev_blocksize = self.chunk_samples
        try:
            with _silence_fd(2):
                self._stream = self._open_stream(sd, dev_sr, dev_blocksize)
        except sd.PortAudioError as exc:
            if "Invalid sample rate" not in str(exc):
                raise
            info = sd.query_devices(
                self.device if self.device is not None else sd.default.device[0],
                kind="input",
            )
            dev_sr = int(info["default_samplerate"])
            dev_blocksize = max(self.chunk_samples, int(self.chunk_samples * dev_sr / self.sr))
            log.info(
                "device refused %d Hz — falling back to native %d Hz + resample",
                self.sr,
                dev_sr,
            )
            self._stream = self._open_stream(sd, dev_sr, dev_blocksize)

        needs_resample = dev_sr != self.sr
        if needs_resample:
            import librosa

        self._stream.start()
        buf = np.zeros(0, dtype=np.float32)
        try:
            while True:
                block = await self._queue.get()
                if needs_resample:
                    block = librosa.resample(
                        block.astype(np.float32),
                        orig_sr=dev_sr,
                        target_sr=self.sr,
                        res_type="polyphase",
                    )
                buf = np.concatenate([buf, block.astype(np.float32)])
                while buf.size >= self.chunk_samples:
                    yield buf[: self.chunk_samples]
                    buf = buf[self.chunk_samples :]
        finally:
            self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class MixedSource:
    """Mix a mic stream with an optional Pulse/Pipewire monitor stream.

    Primary is always a ``MicSource`` (PortAudio). Secondary, if enabled,
    is a ``ParecordSource`` reading from a named Pulse source — typically
    a ``*.monitor`` source for desktop audio. Both yield ``chunk_samples``
    of float32 at ``sr``; the mixer averages paired chunks so the mixed
    signal stays bounded.
    """

    def __init__(
        self,
        sr: int,
        chunk_samples: int,
        primary_device: Optional[int],
        include_secondary: bool = False,
        secondary_pulse_source: Optional[str] = None,
        secondary_gain: float = 1.0,
    ) -> None:
        self.sr = sr
        self.chunk_samples = chunk_samples
        self._primary = MicSource(sr, chunk_samples, device=primary_device)
        # ``secondary_pulse_source=None`` with ``include_secondary=True`` means
        # "let parecord pick the Pulse default source".
        self._secondary: Optional[ParecordSource] = (
            ParecordSource(
                sr, chunk_samples, source_name=secondary_pulse_source, gain=secondary_gain
            )
            if include_secondary
            else None
        )
        self._tasks: list[asyncio.Task] = []

    async def _feed(self, source, q: asyncio.Queue) -> None:
        async for chunk in source.chunks():
            await q.put(chunk)

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Yield mixed (mic + monitor) chunks; falls through to mic-only if no secondary is enabled."""
        if self._secondary is None:
            async for c in self._primary.chunks():
                yield c
            return

        q_a: asyncio.Queue[np.ndarray] = asyncio.Queue()
        q_b: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._tasks = [
            asyncio.create_task(self._feed(self._primary, q_a)),
            asyncio.create_task(self._feed(self._secondary, q_b)),
        ]
        try:
            while True:
                a, b = await asyncio.gather(q_a.get(), q_b.get())
                # Average to keep the mixed signal bounded; both sides are
                # float32 at roughly microphone level already.
                yield ((a.astype(np.float32) + b.astype(np.float32)) * 0.5).astype(np.float32)
        finally:
            self.close()

    def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks = []
        self._primary.close()
        if self._secondary is not None:
            self._secondary.close()


class FileSource:
    """Stream an audio file as fixed-size chunks at ``sr``, optionally pacing in real time."""

    def __init__(
        self,
        path: str | Path,
        sr: int,
        chunk_samples: int,
        realtime: bool = True,
    ) -> None:
        self.path = Path(path)
        self.sr = sr
        self.chunk_samples = chunk_samples
        self.realtime = realtime

    async def chunks(self) -> AsyncIterator[np.ndarray]:
        """Read, resample, and yield float32 mono chunks from the file."""
        import soundfile as sf
        import librosa

        period = self.chunk_samples / self.sr

        with sf.SoundFile(str(self.path)) as f:
            file_sr = f.samplerate
            # Read in large blocks, resample, then slice into chunk_samples.
            # Oversize the read buffer so resampling is stable across boundaries.
            read_block = max(file_sr, self.chunk_samples * 4)
            leftover = np.zeros(0, dtype=np.float32)
            for block in f.blocks(blocksize=read_block, dtype="float32", always_2d=True):
                mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
                if file_sr != self.sr:
                    mono = librosa.resample(
                        mono.astype(np.float32), orig_sr=file_sr, target_sr=self.sr
                    )
                buf = np.concatenate([leftover, mono.astype(np.float32)])
                idx = 0
                while idx + self.chunk_samples <= buf.shape[0]:
                    yield buf[idx : idx + self.chunk_samples]
                    idx += self.chunk_samples
                    if self.realtime:
                        await asyncio.sleep(period)
                    else:
                        await asyncio.sleep(0)
                leftover = buf[idx:]
            # Flush trailing samples (pad to chunk size so final frame gets emitted)
            if leftover.size > 0:
                pad = np.zeros(self.chunk_samples - leftover.size, dtype=np.float32)
                yield np.concatenate([leftover, pad])

    def close(self) -> None:
        pass
