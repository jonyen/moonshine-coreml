#!/usr/bin/env python3
"""Parakeet-tdt_ctc-110m Core ML CTC spike runner.

Loads the OpenVoiceOS Core ML conversion of nvidia/parakeet-tdt_ctc-110m
(mel+FastConformer encoder fused into one mlpackage, taking RAW AUDIO in;
separate CTC head mlpackage), transcribes the repo test clips with CPU-only
compute, compares against the golden .txt transcripts, and times a warm run.

Usage:
    .venv-parakeet/bin/python run_ctc.py                    # full evidence run
    .venv-parakeet/bin/python run_ctc.py file.wav           # one-off transcription
    .venv-parakeet/bin/python run_ctc.py --precision int8   # int8 build (models/int8/)

--precision fp16 (the default) runs the packages in models/ — their weights are
already FLOAT16 despite metadata.json claiming FLOAT32. int8 runs models/int8/,
produced by make_int8.py.
"""
import json
import sys
import time
import wave
from pathlib import Path

import coremltools as ct
import numpy as np

HERE = Path(__file__).parent
MODELS = HERE / "models"
REPO_ROOT = HERE.parent.parent
TEST_ASSETS = REPO_ROOT / "test-assets"
SAMPLE_RATE = 16_000


def load_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono Int16 WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: expected 16 kHz"
        assert w.getsampwidth() == 2, f"{path}: expected Int16"
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels())[:, 0]
    return (data.astype(np.float32) / 32768.0)


def decode_ctc(log_probs: np.ndarray, vocab: list[str], blank_id: int) -> str:
    """Greedy CTC: argmax per frame, collapse repeats, drop blanks."""
    ids = np.argmax(log_probs[0], axis=-1)
    out, prev = [], None
    for t in ids:
        if t != blank_id and t != prev:
            out.append(int(t))
        prev = t
    return "".join(vocab[i] for i in out).replace("▁", " ").strip()


class ParakeetCTC:
    def __init__(self, compute_units=ct.ComputeUnit.CPU_ONLY, precision="fp16"):
        packages = MODELS if precision == "fp16" else MODELS / precision
        self.meta = json.loads((MODELS / "metadata.json").read_text())
        self.vocab = json.loads((MODELS / "vocab.json").read_text())
        self.blank_id = self.meta["blank_id"]
        self.max_samples = self.meta["max_audio_samples"]  # 240000 = 15 s
        self.mel_enc = ct.models.MLModel(
            str(packages / "parakeet_mel_encoder.mlpackage"), compute_units=compute_units)
        self.ctc = ct.models.MLModel(
            str(packages / "parakeet_ctc_decoder.mlpackage"), compute_units=compute_units)

    def transcribe(self, audio: np.ndarray) -> str:
        actual = min(len(audio), self.max_samples)
        padded = np.pad(audio[:actual], (0, self.max_samples - actual))
        enc_out = self.mel_enc.predict({
            "audio_signal": padded.reshape(1, -1).astype(np.float32),
            "audio_length": np.array([actual], dtype=np.int32),
        })
        enc = enc_out["encoder"]
        enc_len = int(enc_out["encoder_length"][0])
        # CTC head expects the fixed 188-frame encoder tensor; slice log-probs after.
        ctc_out = self.ctc.predict({"encoder": enc})
        log_probs = ctc_out["log_probs"][:, :enc_len, :]
        return decode_ctc(log_probs, self.vocab, self.blank_id)


def normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()


def main():
    args = sys.argv[1:]
    precision = "fp16"
    if "--precision" in args:
        i = args.index("--precision")
        precision = args[i + 1]
        del args[i:i + 2]
    model = ParakeetCTC(compute_units=ct.ComputeUnit.CPU_ONLY, precision=precision)

    if args:
        print(model.transcribe(load_wav(Path(args[0]))))
        return

    print(f"Compute units: CPU_ONLY, precision: {precision}\n")
    results = []
    for name in ("hello", "weather", "coffee"):
        wav = TEST_ASSETS / f"{name}.wav"
        golden = (TEST_ASSETS / f"{name}.txt").read_text().strip()
        hyp = model.transcribe(load_wav(wav))
        match = normalize(hyp) == normalize(golden)
        results.append(match)
        print(f"[{name}]")
        print(f"  golden : {golden}")
        print(f"  coreml : {hyp}")
        print(f"  match (case/punct-insensitive): {match}\n")

    # Warm timing on hello.wav (model + audio already loaded, 1 warm-up done above)
    audio = load_wav(TEST_ASSETS / "hello.wav")
    model.transcribe(audio)  # extra warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.transcribe(audio)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"Warm transcription of hello.wav (CPU_ONLY, 5 runs): "
          f"mean {np.mean(times):.1f} ms, min {np.min(times):.1f} ms, "
          f"max {np.max(times):.1f} ms")
    print(f"\nAll clips matched: {all(results)}")


if __name__ == "__main__":
    main()
