"""Local audio file loader for the critical-set tier.

Returns mono float32 numpy arrays at the project's canonical 16 kHz, ready to
hand straight into ``SplitWriter.write`` or any of the labelers in
``labeling.py`` (all of which already accept ``np.ndarray``).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from labeling import SR  # 16 kHz canonical rate

from .crit_config import CritEntry


_SF_NATIVE = {".wav", ".flac", ".ogg"}


def load_recording(entry: CritEntry) -> np.ndarray:
    """Read ``entry.file`` → mono float32 @ ``SR`` (16 kHz)."""
    path = Path(entry.file)
    suffix = path.suffix.lower()

    if suffix in _SF_NATIVE:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            # Mix-down to mono by averaging — same convention as MixedSource etc.
            data = data.mean(axis=1).astype(np.float32, copy=False)
    else:
        # mp3, m4a, opus, etc. — defer to pydub which uses ffmpeg.
        from pydub import AudioSegment

        seg = AudioSegment.from_file(str(path))
        seg = seg.set_channels(1)
        file_sr = seg.frame_rate
        samples = np.asarray(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(2 ** (8 * seg.sample_width - 1))
        data = samples

    if file_sr != SR:
        import librosa
        data = librosa.resample(
            data.astype(np.float32, copy=False),
            orig_sr=file_sr,
            target_sr=SR,
            res_type="polyphase",
        )

    return np.ascontiguousarray(data, dtype=np.float32)
