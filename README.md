# moonshine-coreml

Moonshine Tiny speech-to-text converted to Core ML, plus `MoonshineKit`, a dependency-free Swift
package that runs it on watchOS 11+, iOS 18+ and macOS 15+. See
[`docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md`](docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md)
for the design.

## Models

Compiled models are a GitHub release asset, not part of this git repo (`build/` and `dist/` are
gitignored — the encoder/decoder are large binary `.mlmodelc` bundles that don't belong in git
history). The release asset `moonshine-tiny-coreml-v0.1.0.zip` contains, at its root:

- `Encoder.mlmodelc/`
- `Decoder.mlmodelc/`
- `vocab.json`

Fetch them with:

```bash
Scripts/fetch-models.sh 0.1.0
```

which downloads and unzips the release asset into `build/` (or a directory you pass as a second
argument). See `convert/README.md` if you'd rather produce them yourself from the original weights.

## Using MoonshineKit

```swift
.package(url: "https://github.com/jonyen/moonshine-coreml", from: "0.1.0")
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
`Encoder.mlmodelc`, `Decoder.mlmodelc` and `vocab.json`.

## Bench

```bash
swift run -c release moonshine-bench --models build test-assets/hello.wav
swift run -c release moonshine-bench --models build test-assets/hello.wav --cpu
```

On `test-assets/hello.wav` (3.0 s), M4 Pro, warm:

| computeUnits | warm | RTF  | encode | ms/token |
|--------------|------|------|--------|----------|
| `.all`       | 58 ms | 0.02 | 15 ms | 2.5 |
| `.cpuOnly`   | 25 ms | —    | —      | 1.2 |

Known issue: on this Mac the ANE compiler currently fails non-fatally at load time
(`E5RT ... MILCompilerForANE` / ANECCompile warning) and Core ML silently falls back to another
compute unit, which is why `--cpu` is currently the faster mode on macOS here. On-watch behavior is
measured separately in the watch app project.

## Converting yourself

See [`convert/README.md`](convert/README.md) for converting `UsefulSensors/moonshine-tiny` to Core
ML yourself instead of fetching the release. Parity between the Hugging Face PyTorch reference and
the compiled Core ML models — exact token-for-token match on all three `test-assets/` clips — is
checked with:

```bash
cd convert && uv run python parity_test.py
```

## Tests

```bash
swift test
MOONSHINE_MODELS=$PWD/build swift test
```

The first runs everything that doesn't need the compiled models. Setting `MOONSHINE_MODELS` to a
directory containing `Encoder.mlmodelc`, `Decoder.mlmodelc` and `vocab.json` additionally enables
`TranscriberIntegrationTests`, which run real audio through the models end to end.

## Related

- [jonyen/apple-watch-captions](https://github.com/jonyen/apple-watch-captions) — the watch app integrating this.
- [jonyen/caption-core](https://github.com/jonyen/caption-core)
- [jonyen/mac-live-captions](https://github.com/jonyen/mac-live-captions)

## License

This repo is MIT (see `LICENSE`). The model is `UsefulSensors/moonshine-tiny`, also MIT.
