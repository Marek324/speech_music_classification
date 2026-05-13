# scripts/dataset/crit/crit_label.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

"""Auto-label a single audio file with silero VAD; emit manifest snippet.

Workflow:
    1. Run once on a clip:
         uv run python scripts/dataset/crit/crit_label.py sources/speech/s_whisper.wav
       Writes:
         scripts/dataset/crit/recordings/s_whisper.wav         (16 kHz mono resample)
         scripts/dataset/crit/recordings/s_whisper.labels.txt  (Audacity-format TSV)
       Prints the manifest TOML block to stdout.

    2. (Optional) Edit the labels in Audacity:
         File → Import → Audio… → recordings/s_whisper.wav
         File → Import → Labels… → recordings/s_whisper.labels.txt
         (edit, then) File → Export → Labels… → overwrite the same .labels.txt

    3. Re-run the same command — it reuses the edited labels.txt (no re-VAD)
       and prints an updated manifest TOML block. Pass --force-vad to overwrite
       labels.txt with a fresh silero seed.

Paths are resolved relative to the current working directory; absolute paths work too.
Output filenames default to SPEECH_PATH.stem; pass --out to override.
"""
import argparse
from pathlib import Path

import soundfile as sf

from _common import (
    RECORDINGS_DIR,
    SR,
    load_mono,
    print_manifest,
    read_labels_tsv,
    vad_labels,
    write_labels_tsv,
)


def main() -> None:
    """CLI entry point: VAD-label a speech clip, save resampled wav + labels TSV, print manifest snippet."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("speech_path", type=Path, help="path to the speech audio file")
    p.add_argument("--out", default=None, help="output name (default: SPEECH_PATH.stem)")
    p.add_argument("--subclass", default="clean", help='manifest subclass (default: "clean")')
    p.add_argument("--force-vad", action="store_true", help="re-run VAD even if labels.txt exists")
    args = p.parse_args()

    speech_path: Path = args.speech_path
    if not speech_path.exists():
        raise SystemExit(f"speech file not found: {speech_path}")
    out_name: str = args.out or speech_path.stem

    speech = load_mono(speech_path)
    print(f"speech: {len(speech) / SR:6.2f} s   ({speech_path.name})")

    out_wav = RECORDINGS_DIR / f"{out_name}.wav"
    out_labels = RECORDINGS_DIR / f"{out_name}.labels.txt"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), speech, SR, subtype="PCM_16")
    print(f"wrote → {out_wav}")

    if out_labels.exists() and not args.force_vad:
        labels = read_labels_tsv(out_labels)
        print(f"reused {len(labels)} label(s) from {out_labels.name} (skip VAD)")
    else:
        labels = vad_labels(speech)
        write_labels_tsv(labels, out_labels)
        print(f"silero VAD → {len(labels)} label(s) → {out_labels}")

    notes = f"Speech-only ({speech_path.name}); labelled via silero VAD + manual edits."
    print_manifest(out_name, args.subclass, notes, labels)


if __name__ == "__main__":
    main()
