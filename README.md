# moonshine-coreml

Moonshine speech-to-text (Tiny and Base) converted to Core ML, plus `MoonshineKit`, a
dependency-free Swift package that runs either model on watchOS 11+, iOS 18+ and macOS 15+. See
[`docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md`](docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md)
for the design.

## Models

Two model sizes are converted; both pass exact greedy-transcript parity against the Hugging Face
fp32 reference on all three `test-assets/` clips. Bench numbers are `moonshine-bench --cpu` on
`test-assets/hello.wav` (3.0 s), M4 Pro, warm:

| model | params | zip size | warm | RTF  | ms/token |
|-------|--------|----------|------|------|----------|
| tiny  | 27.1 M | 48 MB    | 25 ms | 0.01 | 1.2 |
| base  | 61.5 M | 108 MB   | 47 ms | 0.02 | 2.5 |

Compiled models are a GitHub release asset, not part of this git repo (`build/`, `build-base/` and
`dist/` are gitignored — the encoder/decoder are large binary `.mlmodelc` bundles that don't belong
in git history). Each release asset `moonshine-<model>-coreml-v<version>.zip` contains, at its root:

- `Encoder.mlmodelc/`
- `Decoder.mlmodelc/`
- `vocab.json`

Fetch them with:

```bash
Scripts/fetch-models.sh 0.2.0        # tiny → build/
Scripts/fetch-models.sh 0.2.0 base   # base → build-base/
```

which downloads and unzips the release asset into `build/` for tiny, `build-base/` for base (or a
directory you pass as a final argument). See `convert/README.md` if you'd rather produce them
yourself from the original weights.

## Parakeet (ParakeetKit)

v0.3.0 adds a second engine behind the same `Transcribing` protocol:
[`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) run down
its CTC path. It is ~4x the size of Moonshine tiny on disk but emits punctuation and
capitalisation, and its mel frontend is baked into the encoder — raw 16 kHz audio in, no DSP in
Swift. The encoder takes a fixed 15 s window (zero-padded, true length passed alongside), so
every transcription pays the full-window encode regardless of clip length.

Both builds transcribe the three `test-assets/` clips character-for-character identically to the
golden transcripts. Bench is `parakeet-bench` on `test-assets/hello.wav` (3.0 s), M4 Pro, warm
mean of 5:

| build | params | size | warm `.cpuOnly` | RTF | warm `.all` |
|-------|--------|------|-----------------|------|-------------|
| fp16 (zip root) | ~103 M shipped (110m checkpoint) | 208 MB | 26 ms | 0.009 | 13 ms |
| int8 (`int8/`)  | same, weights int8 | 105 MB | 26 ms | 0.009 | 13 ms |

The release asset `parakeet-ctc-110m-coreml-v<version>.zip` contains, at its root,
`Encoder.mlmodelc/` (mel + FastConformer encoder), `CTCHead.mlmodelc/`, `vocab.json`
(1024 SentencePiece pieces; CTC blank is 1024) — that root build's weights are fp16, which is
how the upstream conversion ships — plus `int8/`, the same three files with weights
linear-quantized to int8. Fetch with:

```bash
Scripts/fetch-models.sh 0.3.0 parakeet   # → build-parakeet/
```

```swift
import MoonshineKit
import ParakeetKit

let model = try ParakeetModel(directory: modelsDirectory)   // computeUnits: .cpuOnly on the watch
let live = LiveTranscriber(transcriber: ParakeetTranscriber(model: model))
```

`ParakeetTranscriber` conforms to MoonshineKit's `Transcribing`, so the `Segmenter` /
`LiveTranscriber` pipeline (and anything else built on it) takes it unchanged. Bench with
`swift run -c release parakeet-bench --models build-parakeet test-assets/hello.wav --cpu`; run
the integration tests with `PARAKEET_MODELS=$PWD/build-parakeet swift test`. Packaging is
`Scripts/package-models.sh <version> parakeet`, which compiles the mlpackages fetched by
`convert/parakeet/fetch_models.sh` (see [`convert/parakeet/README.md`](convert/parakeet/README.md)).

**License:** the Parakeet model is CC-BY-4.0 — the checkpoint is NVIDIA's
`parakeet-tdt_ctc-110m`, converted to Core ML by
[OpenVoiceOS/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/OpenVoiceOS/parakeet-tdt-ctc-110m-coreml);
this repo repackages that conversion's CTC path (compiled to `.mlmodelc`, plus the int8
quantization) without architectural changes. Attribution to both is required; commercial use is
fine. MoonshineKit/ParakeetKit code stays MIT.

## Using MoonshineKit

```swift
.package(url: "https://github.com/jonyen/moonshine-coreml", from: "0.2.0")
```

```swift
import MoonshineKit

let model = try MoonshineModel(directory: modelsDirectory)
let transcriber = Transcriber(model: model)
let live = LiveTranscriber(transcriber: transcriber)
live.onPartial = { text in print("partial:", text) }
live.onFinal = { text in print("final:", text) }
live.onError = { error in print("error:", error) }

// Feed Int16 16 kHz mono PCM as it arrives from the mic, in whatever chunk size is convenient.
live.feed(samples)
live.flush() // ends the open segment as a final
```

`modelsDirectory` is wherever `Scripts/fetch-models.sh` (or your own build) put
`Encoder.mlmodelc`, `Decoder.mlmodelc` and `vocab.json` — MoonshineKit reads the model's
dimensions off the loaded models, so the same code runs tiny or base.

## Bench

```bash
swift run -c release moonshine-bench --models build test-assets/hello.wav
swift run -c release moonshine-bench --models build test-assets/hello.wav --cpu
swift run -c release moonshine-bench --models build-base test-assets/hello.wav --cpu
```

Tiny on `test-assets/hello.wav` (3.0 s), M4 Pro, warm:

| computeUnits | warm | RTF  | encode | ms/token |
|--------------|------|------|--------|----------|
| `.all`       | 58 ms | 0.02 | 15 ms | 2.5 |
| `.cpuOnly`   | 25 ms | —    | 6 ms   | 1.2 |

Base, same clip and machine, `.cpuOnly`: warm 47 ms, RTF 0.02, encode 10 ms, 2.5 ms/token.

Known issue: on this Mac the ANE compiler currently fails non-fatally at load time
(`E5RT ... MILCompilerForANE` / ANECCompile warning) and Core ML silently falls back to another
compute unit, which is why `--cpu` is currently the faster mode on macOS here. On-watch behavior is
measured separately in the watch app project.

## Converting yourself

See [`convert/README.md`](convert/README.md) for converting `UsefulSensors/moonshine-tiny` or
`UsefulSensors/moonshine-base` to Core ML yourself instead of fetching the release. Parity between
the Hugging Face PyTorch reference and the compiled Core ML models — exact token-for-token match on
all three `test-assets/` clips — is checked with:

```bash
cd convert && uv run python parity_test.py               # tiny, against build/
cd convert && uv run python parity_test.py --model base  # base, against build-base/
```

## Tests

```bash
swift test
MOONSHINE_MODELS=$PWD/build swift test
MOONSHINE_MODELS=$PWD/build-base swift test
PARAKEET_MODELS=$PWD/build-parakeet swift test
```

The first runs everything that doesn't need the compiled models. Setting `MOONSHINE_MODELS` to a
directory containing `Encoder.mlmodelc`, `Decoder.mlmodelc` and `vocab.json` (either model size)
additionally enables `TranscriberIntegrationTests`, which run real audio through the models end to
end. Setting `PARAKEET_MODELS` to a directory containing `Encoder.mlmodelc`, `CTCHead.mlmodelc`
and `vocab.json` (the fp16 root or the `int8/` variant) likewise enables
`ParakeetIntegrationTests`.

## Related

- [jonyen/apple-watch-captions](https://github.com/jonyen/apple-watch-captions) — the watch app integrating this.
- [jonyen/caption-core](https://github.com/jonyen/caption-core)
- [jonyen/mac-live-captions](https://github.com/jonyen/mac-live-captions)

## License

This repo is MIT (see `LICENSE`). The Moonshine models are `UsefulSensors/moonshine-tiny` and
`UsefulSensors/moonshine-base`, also MIT. The Parakeet model is CC-BY-4.0 (NVIDIA checkpoint,
OpenVoiceOS Core ML conversion — see the attribution in the Parakeet section above).
