#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from pydub import AudioSegment


def generate_sine(duration_ms: int, freq_hz: float = 440.0, sample_rate: int = 44100) -> AudioSegment:
    t = np.arange(int(sample_rate * (duration_ms / 1000.0))) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return AudioSegment(
        int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )


def detect_unit(utterances):
    if not utterances:
        return "ms"
    sample_size = min(5, len(utterances))
    max_val = 0.0
    for utt in utterances[:sample_size]:
        start_val = float(utt.get("start", 0))
        end_val = float(utt.get("end", 0))
        max_val = max(max_val, start_val, end_val)
    if max_val >= 1_000_000:
        return "us"
    elif max_val >= 10_000:
        return "ms"
    return "s"


def to_ms(val, unit):
    v = float(val)
    if unit == "us":
        return v / 1000.0
    if unit == "ms":
        return v
    return v * 1000.0


def main():
    out_dir = Path("dataset/mock_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 10 seconds sine
    audio = generate_sine(10_000)
    audio.export(out_dir / "mock_full.wav", format="wav")

    # Mock transcript in microseconds (us)
    transcript = {
        "utterances": [
            {"text": "a", "speaker": "A", "start": 2_880_000, "end": 6_400_000},  # 2.88s-6.4s
            {"text": "b", "speaker": "B", "start": 6_560_000, "end": 8_720_000},  # 6.56s-8.72s
            {"text": "c", "speaker": "C", "start": 9_040_000, "end": 9_680_000},  # 9.04s-9.68s
        ]
    }

    unit = detect_unit(transcript["utterances"])
    print(f"Detected unit: {unit}")

    for i, utt in enumerate(transcript["utterances"]):
        start_ms = to_ms(utt["start"], unit)
        end_ms = to_ms(utt["end"], unit)
        seg = audio[int(round(start_ms)) : int(round(end_ms))]
        out_path = out_dir / f"mock_seg_{i}.wav"
        seg.export(out_path, format="wav")
        print(
            f"seg {i}: start_ms={start_ms:.1f}, end_ms={end_ms:.1f}, duration_ms={len(seg)} -> {out_path}"
        )

    # Also dump a JSON showing normalized ms
    normalized = [
        {
            "start_ms": to_ms(u["start"], unit),
            "end_ms": to_ms(u["end"], unit),
            "text": u["text"],
            "speaker": u["speaker"],
        }
        for u in transcript["utterances"]
    ]
    with open(out_dir / "normalized_transcript_ms.json", "w") as f:
        json.dump({"utterances": normalized}, f, indent=2)


if __name__ == "__main__":
    main()


