# Parakeet-tdt_ctc-110m Core ML spike

**Outcome: WORKS.** `nvidia/parakeet-tdt_ctc-110m` (CC-BY-4.0) runs as Core ML on this Mac
with CPU-only compute and transcribes all three `test-assets/` clips **character-for-character
identically** to the golden transcripts — punctuation and casing included. Warm transcription
of `hello.wav` is ~26 ms end to end (Apple M4 Pro, macOS 26.7, `CPU_ONLY`).

This is a feasibility spike, deliberately separate from the Moonshine pipeline. Nothing under
`Sources/`, `convert/convert.py`, etc. was touched.

## Approach: crib an existing conversion, run the CTC path only

Per the spike guidance, existing public Core ML conversions were surveyed first. Two complete
ones of exactly this checkpoint exist:

| Repo | Notes |
|---|---|
| [OpenVoiceOS/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/OpenVoiceOS/parakeet-tdt-ctc-110m-coreml) | **Used here.** Mel + FastConformer encoder fused into one mlpackage (raw audio in), separate 1 MB CTC head, TDT decoder + joint as separate packages, `vocab.json`, reference `infer.py`. FP32, CoreML7 opset (macOS 14+ / iOS 17+). An [int8 variant](https://huggingface.co/OpenVoiceOS/parakeet-tdt-ctc-110m-coreml-int8) also exists. |
| [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) | Same checkpoint, split as MelSpectrogram / AudioEncoder / CtcHead mlmodelc (~106 MB, likely FP16), with conversion scripts in-repo and a production Swift runtime ([FluidAudio](https://github.com/FluidInference/FluidAudio)). Good second source and the obvious crib for a from-scratch conversion. |

Since ready `.mlpackage`s with the mel front end **baked into the model** existed, converting
from NeMo ourselves would have proven nothing extra for a feasibility question — so this spike
downloads the OpenVoiceOS packages (`fetch_models.sh`) and runs them with `coremltools` from
Python (`run_ctc.py`). No NeMo, no torch; the venv is just `coremltools + numpy +
huggingface_hub` (`.venv-parakeet/`, separate from `convert/.venv`).

Only the CTC head is exercised: greedy CTC decode = argmax per frame, collapse repeats, drop
blanks (`blank_id = 1024`), join SentencePiece pieces, `▁ → space`. The TDT decoder/joint
packages exist in the source repo but are ignored here.

## Evidence

```
$ .venv-parakeet/bin/python run_ctc.py
Compute units: CPU_ONLY

[hello]
  golden : Hello, this is a test of captions running on the watch.
  coreml : Hello, this is a test of captions running on the watch.
  match (case/punct-insensitive): True

[weather]
  golden : The weather tomorrow looks cold, with a chance of rain in the afternoon.
  coreml : The weather tomorrow looks cold, with a chance of rain in the afternoon.
  match (case/punct-insensitive): True

[coffee]
  golden : Could you pick up some coffee and milk on the way home?
  coreml : Could you pick up some coffee and milk on the way home?
  match (case/punct-insensitive): True

Warm transcription of hello.wav (CPU_ONLY, 5 runs): mean 25.7 ms, min 25.4 ms, max 26.5 ms

All clips matched: True
```

All three are **exact** matches (the bar was intelligibility parity modulo case/punct; Parakeet's
punctuation + casing happened to reproduce the goldens verbatim). Per-stage warm timing:
mel+encoder ≈ 24.8 ms (for the full padded 15 s window — cost is length-independent), CTC head
≈ 0.7 ms. Sanity checks: outputs change when the input changes (no result caching), and
`encoder_length` = 38 frames for the 3 s clip (80 ms/frame after 8× subsampling ✓).

## Model facts

- **Size on disk (CTC path):** 217.5 MB — encoder mlpackage 216.5 MB, CTC head 1.1 MB.
  ~~FP32 weights~~ *Correction (productization):* the weights are already **FLOAT16** —
  metadata.json's "FLOAT32" describes the I/O, not the consts (717 of 718 float consts in the
  serialized MIL program are fp16; 206 MB weight.bin ≈ 103 M fp16 params — fp32 would be twice
  that). So an FP16 re-conversion is a no-op; the size reduction that exists from here is int8
  (see the addendum below). Compare Moonshine tiny at ~27M params.
- **I/O contract:** `audio_signal` `(1, 240000)` float32 raw 16 kHz audio (fixed 15 s window,
  zero-padded) + `audio_length` `(1,)` int32 → `encoder` `(1, 512, 188)` + `encoder_length`.
  CTC head: `encoder` → `log_probs` `(1, 188, 1025)`. Slice to `encoder_length` frames before
  decoding.
- **Mel is inside the model** — unlike Moonshine (raw audio) there is no Python-side or
  Swift-side DSP needed at all. This was the biggest integration risk and it is simply gone.
- **Fixed 15 s shape** (not enumerated). Cost is paid for the full window regardless of clip
  length. At 25 ms/window on CPU that is fine; a from-scratch conversion could use enumerated
  shapes (1–12 s) like the Moonshine converter does if the constant cost ever mattered.
- **Vocab:** 1024-entry SentencePiece piece list (`vocab.json`, a plain JSON array; index =
  token id, `▁` marks word starts), blank = 1024. Trivial to ship and decode in Swift — no
  tokenizer library needed for CTC (decode is id→string lookup only).

## What Swift integration would need

1. Load the two mlpackages (compiled to mlmodelc) — same `MLModel` plumbing MoonshineKit
   already has. macOS 14+ / iOS 17+ (CoreML7 opset).
2. Feed a fixed 240 000-sample float32 buffer + true length. **No mel frontend needed.**
3. Greedy CTC decode in Swift: ~20 lines (argmax over 1025, collapse, drop blank, join pieces,
   replace `▁`). Ship `vocab.json` as a bundled resource.
4. Long-form audio: chunk at ≤15 s (ideally on silence) and concatenate — same class of logic
   as any fixed-window model.
5. Optional: re-convert at FP16 or grab the int8 variant to cut 217 MB to ~110/55 MB, and
   benchmark ANE (`.all`) vs CPU; this spike only proves the CPU floor, which already beats
   real time by ~100×.

## Effort estimate for production integration

- **ParakeetKit-style Swift wrapper + CTC decode + tests against these clips: ~1 day.**
- Repackaging (FP16 quantize, or re-run FluidInference's conversion scripts for enumerated
  shapes / naming consistency with this repo's `dist/` layout): 1–2 days.
- Long-form chunking + streaming UX: shared with whatever Moonshine already does; incremental.
- License: CC-BY-4.0 (model and conversions) — attribution required, commercial use fine.

## Productization addendum (v0.3.0)

The spike graduated: `ParakeetKit` (in `Sources/ParakeetKit/`) is the Swift port of
`run_ctc.py` — `ParakeetModel` + `CTCDecoder` + `ParakeetTranscriber`, conforming to
MoonshineKit's `Transcribing` so the live pipeline reuses unchanged. `Scripts/package-models.sh
<version> parakeet` compiles the mlpackages here into `build-parakeet/` and zips the release
asset; see the Parakeet section of the top-level README.

Two findings beyond the spike:

1. **The "fp32" packages were already fp16** (see the corrected size fact above), so the
   planned FLOAT16 re-conversion pass would change nothing and was dropped.
2. **int8 halves the size with exact parity.** `make_int8.py` applies per-channel
   linear-symmetric int8 weight quantization (`coremltools.optimize.coreml`), 216.5 → 109.2 MB
   for the encoder, 1.1 → 0.5 MB for the CTC head. `run_ctc.py --precision int8` reproduces all
   three golden transcripts character-for-character, same as the fp16 originals.

## Files

- `fetch_models.sh` — creates `.venv-parakeet`, downloads the CTC-path mlpackages (~220 MB)
  into `models/` (gitignored).
- `run_ctc.py` — evidence runner: transcribes the three test clips CPU-only, compares to
  goldens, times a warm run. Also works as `run_ctc.py some.wav` for one-offs, and takes
  `--precision int8` to run the quantized build in `models/int8/`.
- `make_int8.py` — produces `models/int8/` from the fetched packages.
