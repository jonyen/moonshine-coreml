"""HF (PyTorch fp32) vs Core ML (fp16) greedy transcripts on test-assets/*.wav.

PASS when the decoded texts match for every clip. Also prints the token ids so
an fp16 near-tie that changes a token without changing the text is visible.
--model tiny|base picks the checkpoint (and the default --models dir).
--write-golden saves the HF text as test-assets/<name>.txt for the Swift tests.
"""
import argparse, sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

MODEL_IDS = {"tiny": "UsefulSensors/moonshine-tiny", "base": "UsefulSensors/moonshine-base"}
SAMPLE_RATE = 16000
MAX_FRAMES = 500
BOS, EOS = 1, 2


def max_new_tokens(samples: int) -> int:
    return min(193, int(samples / SAMPLE_RATE * 6.5) + 2)


def hf_ids(model, proc, audio: np.ndarray) -> list[int]:
    inputs = proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens(len(audio)), do_sample=False, num_beams=1)
    ids = out[0].tolist()
    if ids and ids[0] == BOS:
        ids = ids[1:]
    return [i for i in ids if i != EOS]


def coreml_ids(enc, dec, audio: np.ndarray) -> list[int]:
    seconds = min(12, max(1, int(np.ceil(len(audio) / SAMPLE_RATE))))
    padded_audio = np.zeros((1, seconds * SAMPLE_RATE), dtype=np.float32)
    n = min(len(audio), padded_audio.shape[1])
    padded_audio[0, :n] = audio[:n]
    states = enc.predict({"audio": padded_audio})["encoder_states"]          # [1, F, hidden]
    frames = states.shape[1]
    enc_in = np.zeros((1, MAX_FRAMES, states.shape[2]), dtype=np.float16)
    enc_in[:, :frames] = states
    state = dec.make_state()
    token, ids = BOS, []
    for position in range(max_new_tokens(len(audio))):
        out = dec.predict({"token": np.array([[token]], dtype=np.int32), "encoder_states": enc_in,
                           "frames": np.array([frames], dtype=np.int32),
                           "position": np.array([position], dtype=np.int32)}, state=state)
        token = int(out["next_token"][0])
        if token == EOS:
            break
        ids.append(token)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_IDS), default="tiny")
    ap.add_argument("--models", type=Path, default=None,
                    help="compiled-model dir; defaults to ../build for tiny, ../build-<model> otherwise")
    ap.add_argument("--assets", type=Path, default=Path("../test-assets"))
    ap.add_argument("--write-golden", action="store_true")
    args = ap.parse_args()
    if args.models is None:
        args.models = Path("../build") if args.model == "tiny" else Path(f"../build-{args.model}")

    model_id = MODEL_IDS[args.model]
    model = MoonshineForConditionalGeneration.from_pretrained(model_id).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    enc = ct.models.MLModel(str(args.models / "Encoder.mlpackage"))
    dec = ct.models.MLModel(str(args.models / "Decoder.mlpackage"))

    failures = 0
    for wav in sorted(args.assets.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == SAMPLE_RATE and audio.ndim == 1, (wav, sr, audio.shape)
        ref = hf_ids(model, proc, audio)
        got = coreml_ids(enc, dec, audio)
        ref_text = proc.tokenizer.decode(ref, skip_special_tokens=True).strip()
        got_text = proc.tokenizer.decode(got, skip_special_tokens=True).strip()
        ok = ref_text == got_text
        print(f"{'PASS' if ok else 'FAIL'} {wav.name}")
        print(f"   hf:     {ref_text!r}")
        print(f"   coreml: {got_text!r}")
        if ref != got:
            print(f"   ids differ: hf={ref} coreml={got}")
        if args.write_golden:
            wav.with_suffix(".txt").write_text(ref_text + "\n")
        failures += 0 if ok else 1
    print("all clips match" if failures == 0 else f"{failures} clip(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
