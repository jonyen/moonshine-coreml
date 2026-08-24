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
```

The first runs everything that doesn't need the compiled models. Setting `MOONSHINE_MODELS` to a
directory containing `Encoder.mlmodelc`, `Decoder.mlmodelc` and `vocab.json` (either model size)
additionally enables `TranscriberIntegrationTests`, which run real audio through the models end to
end.

## Related

- [jonyen/apple-watch-captions](https://github.com/jonyen/apple-watch-captions) — the watch app integrating this.
- [jonyen/caption-core](https://github.com/jonyen/caption-core)
- [jonyen/mac-live-captions](https://github.com/jonyen/mac-live-captions)

## License

This repo is MIT (see `LICENSE`). The model is `UsefulSensors/moonshine-tiny`, also MIT.
