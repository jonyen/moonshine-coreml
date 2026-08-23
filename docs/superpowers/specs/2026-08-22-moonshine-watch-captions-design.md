# Moonshine Tiny on the Apple Watch — design

**Date:** 2026-08-22
**Status:** approved

## Goal

Live captions on the Apple Watch with speech-to-text running *on the watch
itself*, using Moonshine Tiny (27M parameters, MIT). No phone, no relay, no
network for the captioning path.

Target hardware: Apple Watch with the S9 chip or newer (Series 9 / 10, Ultra 2+),
which has a 4-core Neural Engine. watchOS 11 or newer.

## Why this shape

- `moonshine-swift` (the official Swift package) ships an xcframework for iOS 15+
  and macOS 13+ only; its core is C++ on ONNX Runtime, and ONNX Runtime
  explicitly does not support watchOS. Porting it is weeks of toolchain work
  with no Neural Engine guarantee. Rejected.
- MLX Swift has no watchOS target. Rejected.
- Core ML runs on watchOS and is the only route to the Neural Engine there.
  Converting the PyTorch model (`UsefulSensors/moonshine-tiny` via
  `transformers`) with `coremltools` and writing a small Swift inference loop is
  the realistic path. **Chosen.**
- The original `moonshine-tiny` is used rather than the newer
  `moonshine-streaming-tiny`: simpler architecture, proven conversion surface.
  Moonshine Tiny is utterance-based, not streaming, so "live" captions come from
  energy-gated segmentation plus re-transcribing the growing segment — the
  same technique Moonshine's own live demo uses. The streaming model is a later
  upgrade, not v1.

## Where the code lives

Two repositories:

1. **`jonyen/moonshine-coreml`** (new, this repo) — model conversion, the
   compiled Core ML models as a release asset, and `MoonshineKit`, an
   engine-agnostic Swift package (tokenizer, encoder/decoder loop, segmenter,
   live transcriber), plus a macOS bench CLI and tests. Reusable by
   `mac-live-captions` later.
2. **`jonyen/apple-watch-captions`** (existing) — a `MoonshineEngine` adapter
   conforming to `CaptionEngine` from `caption-core`, and an "On device" entry
   on the watch home screen. Same `SessionController`, `CaptionStore`,
   `CaptionView`, and mic capture as the Deepgram path, which gives a
   Deepgram-vs-Moonshine A/B on the same watch and mic.

A standalone new watch app was considered and rejected: it would duplicate
~600 lines of watch UI and lose the on-device A/B.

## Part 1 — `moonshine-coreml`

### Layout

```
convert/                   Python (uv venv)
  pyproject.toml
  convert.py               HF model → Encoder.mlpackage + Decoder.mlpackage (fp16)
  export_vocab.py          tokenizer.json → vocab.json (32768 strings)
  parity_test.py           HF generate vs Core ML on test wavs
Package.swift              MoonshineKit (library), moonshine-bench (macOS executable), tests
                           platforms: watchOS 11, iOS 18, macOS 15
Sources/MoonshineKit/
  MoonshineModel.swift     loads compiled .mlmodelc from a directory URL; encode() / decodeStep()
  TokenDecoder.swift       vocab.json → text
  Transcriber.swift        actor: [Int16] → String
  Segmenter.swift          pure state machine, no Core ML
  LiveTranscriber.swift    Segmenter + Transcriber glue; onPartial / onFinal
Sources/moonshine-bench/   CLI: `moonshine-bench --models DIR file.wav [--live]`
Tests/MoonshineKitTests/
Scripts/fetch-models.sh    downloads a release zip of compiled models
test-assets/               2–3 short wavs with expected transcripts
docs/superpowers/specs/    this document
```

### Model facts (from the HF config)

`hidden_size` 288, 6 encoder layers, 6 decoder layers, 8 heads, head dim 36
(HF pads it to 40 inside its attention kernel for alignment only — the
padding is zeros, so dot products are unchanged and the converted model keeps
36), vocab 32768,
`max_position_embeddings` 194, `decoder_start_token_id` / bos 1, eos 2,
RoPE with `partial_rotary_factor` 0.9. Encoder input is raw 16 kHz float audio
in [-1, 1]. Tokenizer is Llama-style BPE with `▁` word markers and `<0xNN>`
byte-fallback tokens; ids 0–2 and 32000–32534 are special.

### Conversion (`convert.py`)

Load `UsefulSensors/moonshine-tiny` with `transformers`, wrap the encoder and
decoder in small `torch.nn.Module`s, trace, and convert with `coremltools` to
fp16 ML programs with minimum deployment target watchOS 11 / iOS 18 / macOS 15.

- **`Encoder.mlpackage`** — input `audio` float32 `[1, T]`. Enumerated shapes,
  T ∈ {1, 2, …, 12} s × 16000. The Neural Engine handles enumerated shapes
  well and flexible `RangeDim` shapes poorly; callers zero-pad audio to the next
  bucket. Output `encoder_states` `[1, F, 288]`.
- **`Decoder.mlpackage`** — stateful (`ct.StateType`): self-attention KV cache
  states `k_cache` and `v_cache`, each `[6 layers, 1, 8, 194, 36]` fp16. Every
  shape is fixed, which is what the Neural Engine likes best. Inputs: `token`
  int32 `[1, 1]`; `encoder_states` fp16 `[1, 500, 288]` — the encoder output
  zero-padded to 500 frames (12 s of audio is 498 frames); `frames` int32 `[1]`
  (how many of the 500 are real); `position` int32 `[1]` (index of `token` in
  the sequence, 0 for bos). The causal mask, the encoder padding mask, and the
  one-hot cache write are all derived from `frames` and `position` inside the
  model, so there are no dynamic slices. Output `next_token` int32 `[1]` — the
  argmax is computed inside the model so 32768 logits never cross the Core ML
  boundary per step. Cross-attention K/V are recomputed every step in v1; at
  this size it is cheap, and precomputing them is an optimisation to take only
  if profiling says so.
- **`export_vocab.py`** — `vocab.json`, an array of 32768 token strings.
- After conversion, `xcrun coremlcompiler compile` produces `Encoder.mlmodelc`
  and `Decoder.mlmodelc`. Those two directories plus `vocab.json` are zipped as
  the GitHub release asset `moonshine-tiny-coreml-v<version>.zip` (~55 MB).
  Models are **not** committed to git.
- **`parity_test.py`** — runs HF `generate` and the Core ML pair on the
  test-asset wavs and asserts identical token ids. It also asserts that the
  text for zero-padded audio equals the text for unpadded audio; if that ever
  fails, the encoder switches to `RangeDim` and the enumerated-shape decision is
  revisited.

### MoonshineKit

- **`MoonshineModel`** — `init(directory: URL)` loads the two `.mlmodelc`
  bundles with `MLModelConfiguration.computeUnits = .all` and `vocab.json`.
  `encode(_ audio: [Float]) throws -> MLMultiArray` pads to the bucket and runs
  the encoder. `decodeStep(token:encoderStates:position:state:)` runs one
  decoder step and returns the next token id. Loading is slow (1–2 s); callers
  keep the instance alive.
- **`TokenDecoder`** — `init(vocab: [String])`, `decode(_ ids: [Int32]) ->
  String`. Drops special ids, turns `▁` into a space, accumulates `<0xNN>`
  byte-fallback tokens into bytes and decodes them as UTF-8, trims leading
  space.
- **`Transcriber`** — an actor. `transcribe(_ samples: [Int16]) async throws ->
  String`: Int16 → Float / 32768, encode, greedy decode from bos until eos or
  `maxTokens = min(193, Int(seconds × 6.5) + 2)` generated tokens (194
  positions including bos), then `TokenDecoder`. A fresh decoder `MLState` per
  utterance. One model instance, one request at a time.
- **`Segmenter`** — a value type fed `[Int16]` chunks; returns events.
  Closed until chunk RMS ≥ 50 (`EnergyGate` logic ported from
  `mac-live-captions`, including the 300 ms pre-roll so the utterance onset is
  not lost). While open it emits `.interim(audio)` at most every 750 ms, and
  `.final(audio)` after 700 ms below threshold or at a 12 s hard cap (after
  which a fresh segment starts immediately). Deterministic and unit-tested
  without Core ML. Thresholds are `init` parameters.
- **`LiveTranscriber`** — owns a `Segmenter` and a `Transcriber`. `feed(_
  samples: [Int16])` on the audio thread; inference runs serially on its own
  task. If inference is busy when an interim is due, the latest audio wins.
  Every result carries its segment id; a partial whose segment has already
  closed is dropped. If segments back up (sustained real-time factor > 1) it
  stops requesting interims and only transcribes finals, and logs that.
  Callbacks: `onPartial: (String) -> Void`, `onFinal: (String) -> Void`.
  `flush()` finalises an open segment (used on stop). Errors from one segment
  are reported via `onError` and do not stop the next segment.

### Bench CLI

`moonshine-bench --models DIR file.wav` prints the transcript, encode time,
mean time per decoded token, and real-time factor. `--live` pushes the file
through `LiveTranscriber` in 100 ms chunks and prints partials/finals with
timestamps, which exercises the segmenter end to end.

### Tests

- `swift test` on macOS: `TokenDecoderTests` (▁, byte fallback, specials),
  `SegmenterTests` (open/preroll/interim cadence/close/hard cap),
  `LiveTranscriberTests` with a fake `Transcribing` protocol (coalescing,
  stale-partial dropping, flush), and an integration test that transcribes the
  test-asset wavs with the real models — skipped unless `MOONSHINE_MODELS`
  points at a models directory.
- `python convert/parity_test.py` — conversion correctness.
- Everything is proven on the Mac before any watch work starts.

## Part 2 — `apple-watch-captions` integration

### Project changes

- `watch/project.yml`: add package `MoonshineKit`
  (`https://github.com/jonyen/moonshine-coreml`, from 0.1.0); raise the
  watchOS deployment target from 10.0 to **11.0** (stateful Core ML models).
- Models live in `watch/Models/MoonshineTiny/` (`Encoder.mlmodelc`,
  `Decoder.mlmodelc`, `vocab.json`), fetched by
  `watch/Scripts/fetch-moonshine.sh <version>` from the `moonshine-coreml`
  release, gitignored like `Secrets.swift`, and bundled as a folder resource.
  The watch README documents the fetch step. The app grows by ~55 MB.

### `MoonshineEngine`

`watch/WatchCaptions/MoonshineEngine.swift`, `final class MoonshineEngine:
CaptionEngine` (the protocol is `start()`, `send(Data)`, `close()`, `onEvent`,
`onClose`).

- `start()` — load the models off the main actor; on success emit `.ready`,
  on failure emit `.error("On-device model failed to load")`. Loaded models are
  kept for the life of the app, so only the first session pays the load.
- `send(_ audio: Data)` — 16 kHz mono Int16 PCM, which is exactly what the
  existing `AudioCapture` emits and exactly what Moonshine wants; forwarded to
  `LiveTranscriber.feed`.
- Partial → `.caption(text:isFinal:false, channel:nil)`; final →
  `.caption(text:isFinal:true, channel:nil)`. `CaptionStore` and `CaptionView`
  are untouched.
- `close()` — drop any open segment and queued work, keep the models.
  (`SessionController.stop()` clears `running` before calling `close()`, so a
  final flushed here would be discarded anyway; `LiveTranscriber.flush()`
  exists for callers that do want it.)
- An inference error drops that segment and the session continues; three
  consecutive failures emit `.error("On-device captions failed")`, which makes
  `SessionController` stop the session as it does today.

### Entry point

A new row on `HomeView` — **"On device"**, `systemImage: "cpu"` — directly under
"Off the record". `AppModel` gains a third `SessionController` with its own
`AudioCapture()`, a `MoonshineEngine`, the shared `store`, and the same
`MicPermission`; `startOnDevice()` pushes the `.captions` screen with an
indicator reading "On device, not saved". In v1 on-device sessions are
live-only, not saved to the relay; saving them is a follow-up. Stop works as
it does for every other session.

### Performance

- Targets on the S9 Neural Engine: encoding 5 s of audio ≤ 300 ms, ≤ 15 ms per
  decoded token — so a 5 s utterance (~30 tokens) finishes in under a second.
  The 750 ms interim cadence self-throttles if inference is slower.
- Measured first with `moonshine-bench` on the Mac (correctness and rough
  timing), then on the watch with `os_signpost` around encode and each decode
  step, read in Instruments.
- If Core ML falls back from the Neural Engine to CPU (an unsupported op in the
  stateful decoder, say), the app still works, only slower; the bench shows it,
  and the options are int8 palettisation or switching the encoder to
  `RangeDim`.

### Verification

- MoonshineKit: `swift test` and the Python parity test on the Mac.
- Watch: `cd watch && xcodegen generate` then
  `xcodebuild -project watch/WatchCaptions.xcodeproj -scheme WatchCaptions
  -destination 'generic/platform=watchOS Simulator' CODE_SIGNING_ALLOWED=NO`
  (simulator runs Core ML on CPU — correctness only), then on the real watch:
  start an "On device" session, speak, captions appear; run a Deepgram session
  on the same watch and compare.

## Out of scope for v1

Saving on-device transcripts to the relay; wiring MoonshineKit into the iOS
or Mac apps (the package supports those platforms, nothing consumes it yet);
the `moonshine-streaming-tiny` model; int8 quantisation unless performance
forces it.
