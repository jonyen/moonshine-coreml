# moonshine-coreml Implementation Plan (Part A of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Moonshine Tiny to Core ML and ship `MoonshineKit`, a Swift package that turns 16 kHz Int16 audio into live partial/final captions on watchOS 11+, proven on the Mac with tests and a bench CLI.

**Architecture:** Python (`coremltools`) converts the HF PyTorch model into an enumerated-shape encoder and a fixed-shape stateful decoder (argmax inside the model). Swift `MoonshineKit` wraps the two models (`MoonshineModel`), runs greedy decoding (`Transcriber`), cuts mic audio into utterances (`Segmenter`), and glues them into a serial live pipeline (`LiveTranscriber`). Compiled models ship as a GitHub release zip, never in git.

**Tech Stack:** Swift 5 language mode / swift-tools 6.0, Core ML (`MLState`, watchOS 11 / iOS 18 / macOS 15), XCTest; Python 3.11 via `uv`, `torch`, `transformers` 4.x, `coremltools` 8+, `soundfile`; macOS `say` + `afconvert` for test audio; `gh` for the release.

**Spec:** `docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md` (this repo). Part B (the watch app integration) is `~/Projects/apple-watch-captions/docs/superpowers/plans/2026-08-22-on-device-moonshine.md` and depends on this plan's release.

## Global Constraints

- Repo: `~/Projects/moonshine-coreml`, public GitHub `jonyen/moonshine-coreml`, MIT.
- Package platforms: `watchOS 11`, `iOS 18`, `macOS 15`. Swift language mode 5.
- MoonshineKit has **no** dependencies (no caption-core, no swift-transformers).
- Audio everywhere is 16 kHz mono Int16 PCM; the model takes Float in [-1, 1] (`Int16 / 32768`), `do_normalize` is false.
- Model: `UsefulSensors/moonshine-tiny`; bos = 1, eos = 2, vocab 32768, 194 positions, hidden 288, 6+6 layers, 8 heads, head dim 36.
- Encoder buckets: 1…12 s; decoder `encoder_states` fixed `[1, 500, 288]` fp16; states `k_cache`/`v_cache` `[6, 1, 8, 194, 36]` fp16.
- Generated-token cap: `min(193, Int(seconds × 6.5) + 2)`.
- Segmenter defaults (samples): threshold RMS 50, preroll 4 800, interim every 12 000, close after 11 200 of silence, hard cap 192 000.
- Compiled models + `vocab.json` are release asset `moonshine-tiny-coreml-v<version>.zip` with `Encoder.mlmodelc`, `Decoder.mlmodelc`, `vocab.json` at the zip root. Not committed. `build/`, `dist/`, `.build/`, `.superpowers/` are gitignored.
- Commit after every task. Commit messages: conventional (`feat:`, `test:`, `docs:`, `chore:`), body in normal prose, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## File map

| Path | Responsibility |
|---|---|
| `Package.swift` | MoonshineKit library, `moonshine-bench` executable, test target |
| `Sources/MoonshineKit/TokenDecoder.swift` | token ids → text |
| `Sources/MoonshineKit/Segmenter.swift` | PCM stream → utterance events (pure) |
| `Sources/MoonshineKit/LiveTranscriber.swift` | Segmenter + Transcribing → partial/final callbacks |
| `Sources/MoonshineKit/MoonshineModel.swift` | loads `.mlmodelc`s, `encode`, `decodeStep` |
| `Sources/MoonshineKit/Transcriber.swift` | `[Int16]` → `String` (greedy loop) |
| `Sources/MoonshineKit/WAVReader.swift` | 16-bit mono 16 kHz WAV → `[Int16]` (bench + tests) |
| `Sources/moonshine-bench/main.swift` | CLI: transcript + timings, `--live` mode |
| `Tests/MoonshineKitTests/*.swift` | unit tests + integration test (env-gated) |
| `convert/pyproject.toml`, `convert/export_vocab.py`, `convert/convert.py`, `convert/parity_test.py` | conversion + verification |
| `Scripts/make-test-assets.sh` | TTS → `test-assets/*.wav` |
| `Scripts/package-models.sh`, `Scripts/fetch-models.sh` | release zip out / in |
| `test-assets/<name>.wav`, `<name>.txt` (golden), `<name>.say.txt` (spoken) | fixtures |
| `README.md`, `LICENSE`, `.gitignore` | docs |

---

### Task A1: Package scaffold, test assets, TokenDecoder

**Files:**
- Create: `Package.swift`, `.gitignore`, `LICENSE`, `README.md` (stub), `Scripts/make-test-assets.sh`, `test-assets/*.wav`, `test-assets/*.say.txt`
- Create: `Sources/MoonshineKit/TokenDecoder.swift`
- Create: `Sources/moonshine-bench/main.swift` (placeholder that prints usage; real CLI in A7)
- Test: `Tests/MoonshineKitTests/TokenDecoderTests.swift`

**Interfaces:**
- Produces: `public struct TokenDecoder { init(vocab: [String]); init(vocabURL: URL) throws; func decode(_ ids: [Int32]) -> String; static let bos: Int32 = 1; static let eos: Int32 = 2 }`

- [ ] **Step 1: Scaffold**

`Package.swift`:
```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MoonshineKit",
    platforms: [.watchOS(.v11), .iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "MoonshineKit", targets: ["MoonshineKit"]),
        .executable(name: "moonshine-bench", targets: ["moonshine-bench"]),
    ],
    targets: [
        .target(name: "MoonshineKit"),
        .executableTarget(name: "moonshine-bench", dependencies: ["MoonshineKit"]),
        .testTarget(name: "MoonshineKitTests", dependencies: ["MoonshineKit"]),
    ],
    swiftLanguageModes: [.v5]
)
```

`.gitignore`:
```
.build/
.swiftpm/
build/
dist/
.superpowers/
convert/.venv/
convert/__pycache__/
*.mlpackage
*.mlmodelc
.DS_Store
```

`LICENSE`: MIT, `Copyright (c) 2026 Jonathan Yen`.

`README.md` stub: title `# moonshine-coreml`, one paragraph: "Moonshine Tiny speech-to-text converted to Core ML, plus `MoonshineKit`, a dependency-free Swift package that runs it on watchOS 11+, iOS 18+ and macOS 15+. See `docs/superpowers/specs/` for the design." (Filled in during A8.)

`Sources/moonshine-bench/main.swift` placeholder:
```swift
import Foundation
print("moonshine-bench: not implemented yet")
```

- [ ] **Step 2: Test assets**

`Scripts/make-test-assets.sh`:
```bash
#!/bin/bash
# Synthesises the short clips the tests and the parity check run on.
# macOS only: uses the system voice. Re-running overwrites the wavs.
set -euo pipefail
cd "$(dirname "$0")/../test-assets"
make() {
  local name="$1" text="$2"
  say -o "tmp.aiff" "$text"
  afconvert -f WAVE -d LEI16@16000 -c 1 tmp.aiff "$name.wav"
  rm tmp.aiff
  printf '%s\n' "$text" > "$name.say.txt"
}
make hello   "Hello, this is a test of captions running on the watch."
make weather "The weather tomorrow looks cold, with a chance of rain in the afternoon."
make coffee  "Could you pick up some coffee and milk on the way home?"
echo "wrote: $(ls *.wav | tr '\n' ' ')"
```
Run: `mkdir -p test-assets && chmod +x Scripts/make-test-assets.sh && Scripts/make-test-assets.sh`
Expected: three wavs (each roughly 100–160 KB), three `.say.txt`. Verify format: `afinfo test-assets/hello.wav | grep -E "16000 Hz|1 ch|16-bit"`.

- [ ] **Step 3: Failing TokenDecoder tests**

`Tests/MoonshineKitTests/TokenDecoderTests.swift`:
```swift
import XCTest
@testable import MoonshineKit

final class TokenDecoderTests: XCTestCase {
    /// A stand-in vocabulary: the decoder only cares about the strings, not the real ids.
    private func makeDecoder() -> TokenDecoder {
        var vocab = Array(repeating: "", count: 32_768)
        vocab[0] = "<unk>"; vocab[1] = "<s>"; vocab[2] = "</s>"
        vocab[10] = "▁Hello"; vocab[11] = "▁world"; vocab[12] = "."; vocab[13] = "▁"
        vocab[20] = "<0xE2>"; vocab[21] = "<0x9C>"; vocab[22] = "<0x93>"
        vocab[32_000] = "<<ST_0>>"
        return TokenDecoder(vocab: vocab)
    }

    func testJoinsWordPiecesAndTrimsLeadingSpace() {
        XCTAssertEqual(makeDecoder().decode([10, 11, 12]), "Hello world.")
    }

    func testSkipsSpecialTokens() {
        XCTAssertEqual(makeDecoder().decode([1, 10, 32_000, 2]), "Hello")
    }

    func testByteFallbackAssemblesUTF8() {
        XCTAssertEqual(makeDecoder().decode([10, 13, 20, 21, 22]), "Hello ✓")
    }

    func testEmptyInputIsEmptyString() {
        XCTAssertEqual(makeDecoder().decode([]), "")
    }

    func testOutOfRangeIdIsIgnored() {
        XCTAssertEqual(makeDecoder().decode([10, 40_000]), "Hello")
    }
}
```
Run: `swift test --filter TokenDecoderTests`
Expected: build failure, `cannot find 'TokenDecoder' in scope`.

- [ ] **Step 4: Implement**

`Sources/MoonshineKit/TokenDecoder.swift`:
```swift
import Foundation

/// Turns Moonshine token ids back into text. The vocabulary is the 32768
/// strings exported from the Hugging Face `tokenizer.json` (Llama-style BPE:
/// "▁" marks a word boundary, "<0xNN>" is one raw UTF-8 byte).
public struct TokenDecoder {
    public static let bos: Int32 = 1
    public static let eos: Int32 = 2
    /// Ids at or above this are the `<<ST_n>>` stream tokens; 0...2 are
    /// `<unk>`, `<s>`, `</s>`. None of them is text.
    public static let firstSpecialID: Int32 = 32_000

    public let vocab: [String]

    public init(vocab: [String]) {
        self.vocab = vocab
    }

    /// Reads a `vocab.json` array of strings.
    public init(vocabURL: URL) throws {
        vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: vocabURL))
    }

    public func decode(_ ids: [Int32]) -> String {
        var out = ""
        var bytes: [UInt8] = []
        func flushBytes() {
            guard !bytes.isEmpty else { return }
            out += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll()
        }
        for id in ids {
            guard id > Self.eos, id < Self.firstSpecialID, Int(id) < vocab.count else { continue }
            let token = vocab[Int(id)]
            if let byte = Self.byteFallback(token) {
                bytes.append(byte)
                continue
            }
            flushBytes()
            out += token.replacingOccurrences(of: "▁", with: " ")
        }
        flushBytes()
        return out.hasPrefix(" ") ? String(out.dropFirst()) : out
    }

    /// `"<0x0A>"` → `0x0A`; nil for any other token.
    static func byteFallback(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }
}
```

- [ ] **Step 5: Run tests**

Run: `swift test --filter TokenDecoderTests`
Expected: `Executed 5 tests, with 0 failures`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: package scaffold, test clips and TokenDecoder"
```

---

### Task A2: Segmenter

**Files:**
- Create: `Sources/MoonshineKit/Segmenter.swift`
- Test: `Tests/MoonshineKitTests/SegmenterTests.swift`

**Interfaces:**
- Produces:
```swift
public struct Segmenter {
    public enum Event: Equatable {
        case interim(audio: [Int16], segment: Int)
        case final(audio: [Int16], segment: Int)
    }
    public init(threshold: Double = 50, prerollSamples: Int = 4_800, interimInterval: Int = 12_000,
                silenceToClose: Int = 11_200, maxSegmentSamples: Int = 192_000)
    public private(set) var isOpen: Bool
    public private(set) var currentSegment: Int   // -1 before the first segment
    public mutating func feed(_ samples: [Int16]) -> [Event]
    public mutating func flush() -> Event?
}
```

- [ ] **Step 1: Failing tests**

`Tests/MoonshineKitTests/SegmenterTests.swift`:
```swift
import XCTest
@testable import MoonshineKit

final class SegmenterTests: XCTestCase {
    /// 100 ms at 16 kHz — the size AudioCapture's tap delivers.
    private let chunk = 1_600
    private func silence() -> [Int16] { Array(repeating: 0, count: chunk) }
    private func speech() -> [Int16] { Array(repeating: 1_000, count: chunk) }

    /// Feeds chunks until the predicate matches an event; returns the events of that chunk and how many chunks it took.
    private func feed(_ s: inout Segmenter, _ make: () -> [Int16], until matches: (Segmenter.Event) -> Bool,
                      limit: Int = 200) -> (events: [Segmenter.Event], chunks: Int) {
        for n in 1...limit {
            let events = s.feed(make())
            if events.contains(where: matches) { return (events, n) }
        }
        XCTFail("no matching event within \(limit) chunks")
        return ([], limit)
    }
    private func isInterim(_ e: Segmenter.Event) -> Bool { if case .interim = e { return true } else { return false } }
    private func isFinal(_ e: Segmenter.Event) -> Bool { if case .final = e { return true } else { return false } }

    func testSilenceProducesNothingAndStaysClosed() {
        var s = Segmenter()
        for _ in 0..<50 { XCTAssertEqual(s.feed(silence()), []) }
        XCTAssertFalse(s.isOpen)
        XCTAssertEqual(s.currentSegment, -1)
    }

    func testOpensWithPrerollAndEmitsFirstInterimAfterInterval() {
        var s = Segmenter()
        for _ in 0..<5 { _ = s.feed(silence()) }
        let (events, chunks) = feed(&s, speech, until: isInterim)
        // Pre-roll holds the last 4_800 samples (3 chunks): 2 of silence + the opening speech chunk.
        // 4_800 + 5 more speech chunks = 12_800 ≥ 12_000, so the interim lands on the 6th speech chunk.
        XCTAssertEqual(chunks, 6)
        XCTAssertTrue(s.isOpen)
        XCTAssertEqual(s.currentSegment, 0)
        guard case .interim(let audio, let segment)? = events.first else { return XCTFail("\(events)") }
        XCTAssertEqual(segment, 0)
        XCTAssertEqual(audio.count, 12_800)
        XCTAssertEqual(Array(audio[0..<3_200]), Array(repeating: 0, count: 3_200))
        XCTAssertEqual(audio[3_200], 1_000)
    }

    func testInterimsKeepComingEveryInterval() {
        var s = Segmenter()
        _ = feed(&s, speech, until: isInterim)
        let (_, chunks) = feed(&s, speech, until: isInterim)
        XCTAssertEqual(chunks, 8)   // 12_000 / 1_600 rounds up to 8 chunks
    }

    func testClosesAfterSilenceAndReArms() {
        var s = Segmenter()
        for _ in 0..<5 { _ = s.feed(silence()) }         // fills the 4_800-sample pre-roll
        for _ in 0..<3 { _ = s.feed(speech()) }          // opens on the first; 4_800 + 2 × 1_600 buffered
        let (events, chunks) = feed(&s, silence, until: isFinal)
        XCTAssertEqual(chunks, 7)                          // 7 × 1_600 = 11_200 of silence
        guard case .final(let audio, let segment)? = events.first else { return XCTFail("\(events)") }
        XCTAssertEqual(segment, 0)
        XCTAssertEqual(audio.count, 4_800 + 2 * 1_600 + 7 * 1_600)
        XCTAssertFalse(s.isOpen)
        XCTAssertEqual(s.feed(silence()), [])
        // The next utterance is a new segment.
        let (next, _) = feed(&s, speech, until: isInterim)
        guard case .interim(_, let nextSegment)? = next.first else { return XCTFail("\(next)") }
        XCTAssertEqual(nextSegment, 1)
    }

    func testHardCapFinalisesAndKeepsGoingInANewSegment() {
        var s = Segmenter()   // no pre-roll: the first chunk opens with just itself, so 120 × 1_600 = 192_000
        let (events, _) = feed(&s, speech, until: isFinal)
        guard case .final(let audio, let segment)? = events.first(where: isFinal) else { return XCTFail("\(events)") }
        XCTAssertEqual(segment, 0)
        XCTAssertEqual(audio.count, 192_000)
        XCTAssertTrue(s.isOpen, "speech is still going; the next segment starts at once")
        XCTAssertEqual(s.currentSegment, 1)
        let (next, _) = feed(&s, speech, until: isInterim)
        guard case .interim(_, let nextSegment)? = next.first else { return XCTFail("\(next)") }
        XCTAssertEqual(nextSegment, 1)
    }

    func testFlushReturnsFinalOnlyWhenOpen() {
        var s = Segmenter()
        XCTAssertNil(s.flush())
        for _ in 0..<5 { _ = s.feed(silence()) }         // fills the pre-roll
        for _ in 0..<3 { _ = s.feed(speech()) }
        guard case .final(let audio, let segment)? = s.flush() else { return XCTFail("expected final") }
        XCTAssertEqual(segment, 0)
        XCTAssertEqual(audio.count, 4_800 + 2 * 1_600)
        XCTAssertFalse(s.isOpen)
        XCTAssertNil(s.flush())
    }

    func testCustomThresholdIsHonoured() {
        var s = Segmenter(threshold: 2_000)
        for _ in 0..<20 { XCTAssertEqual(s.feed(speech()), []) }  // RMS 1_000 < 2_000
        XCTAssertFalse(s.isOpen)
    }
}
```
Run: `swift test --filter SegmenterTests` → build failure, `cannot find 'Segmenter'`.

- [ ] **Step 2: Implement**

`Sources/MoonshineKit/Segmenter.swift`:
```swift
import Foundation

/// Cuts a stream of 16 kHz mono Int16 PCM into utterances worth transcribing.
///
/// Moonshine transcribes whole utterances, so "live" captions come from
/// transcribing the utterance-so-far every `interimInterval` (a gray partial)
/// and once more when it ends (the final). Time is counted in samples, never
/// wall-clock, so this is deterministic and testable without a clock.
///
/// Closed until a chunk's RMS reaches `threshold`; the last `prerollSamples`
/// of silence travel with the opening chunk so the onset is not clipped (the
/// `EnergyGate` idea from mac-live-captions). Open until `silenceToClose`
/// consecutive quiet samples, or `maxSegmentSamples` — the encoder's longest
/// bucket — at which point the segment is finalised and a new one starts
/// immediately because speech is still going.
public struct Segmenter {
    public enum Event: Equatable {
        /// The utterance so far; transcribe for a partial caption.
        case interim(audio: [Int16], segment: Int)
        /// The utterance is over; transcribe for a final caption.
        case final(audio: [Int16], segment: Int)
    }

    public let threshold: Double
    public let prerollSamples: Int
    public let interimInterval: Int
    public let silenceToClose: Int
    public let maxSegmentSamples: Int

    public private(set) var isOpen = false
    /// Id of the open (or most recent) segment; -1 before the first.
    public private(set) var currentSegment = -1

    private var preroll: [Int16] = []
    private var buffer: [Int16] = []
    private var samplesSinceInterim = 0
    private var silentSamples = 0
    private var nextSegment = 0

    public init(threshold: Double = 50,
                prerollSamples: Int = 4_800,        // 300 ms
                interimInterval: Int = 12_000,      // 750 ms
                silenceToClose: Int = 11_200,       // 700 ms
                maxSegmentSamples: Int = 192_000) { // 12 s
        self.threshold = threshold
        self.prerollSamples = prerollSamples
        self.interimInterval = interimInterval
        self.silenceToClose = silenceToClose
        self.maxSegmentSamples = maxSegmentSamples
    }

    public mutating func feed(_ samples: [Int16]) -> [Event] {
        guard !samples.isEmpty else { return [] }
        let loud = Self.rms(samples) >= threshold

        if !isOpen {
            preroll.append(contentsOf: samples)
            if preroll.count > prerollSamples {
                preroll.removeFirst(preroll.count - prerollSamples)
            }
            guard loud else { return [] }
            open(with: preroll)
            preroll = []
            return []
        }

        buffer.append(contentsOf: samples)
        samplesSinceInterim += samples.count
        silentSamples = loud ? 0 : silentSamples + samples.count

        if silentSamples >= silenceToClose {
            let event = Event.final(audio: buffer, segment: currentSegment)
            close()
            return [event]
        }
        if buffer.count >= maxSegmentSamples {
            let event = Event.final(audio: buffer, segment: currentSegment)
            open(with: [])   // no pre-roll: everything so far was just emitted
            return [event]
        }
        if samplesSinceInterim >= interimInterval {
            samplesSinceInterim = 0
            return [.interim(audio: buffer, segment: currentSegment)]
        }
        return []
    }

    /// Ends an open segment (on stop). Returns its final, or nil when closed.
    public mutating func flush() -> Event? {
        guard isOpen else { return nil }
        let event = Event.final(audio: buffer, segment: currentSegment)
        close()
        return event
    }

    private mutating func open(with onset: [Int16]) {
        isOpen = true
        currentSegment = nextSegment
        nextSegment += 1
        buffer = onset
        samplesSinceInterim = onset.count
        silentSamples = 0
    }

    private mutating func close() {
        isOpen = false
        buffer = []
        preroll = []
        samplesSinceInterim = 0
        silentSamples = 0
    }

    static func rms(_ samples: [Int16]) -> Double {
        var sumSquares = 0.0
        for sample in samples { sumSquares += Double(sample) * Double(sample) }
        return (sumSquares / Double(samples.count)).squareRoot()
    }
}
```

- [ ] **Step 3: Run tests**

Run: `swift test --filter SegmenterTests`
Expected: `Executed 7 tests, with 0 failures`. If `testInterimsKeepComingEveryInterval` reports 7 or 9 instead of 8, the count is wrong in the implementation (check `samplesSinceInterim = 0` only when an interim is emitted).

- [ ] **Step 4: Commit**

```bash
git add Sources/MoonshineKit/Segmenter.swift Tests/MoonshineKitTests/SegmenterTests.swift
git commit -m "feat: Segmenter cuts PCM into utterances with interim ticks"
```

---

### Task A3: LiveTranscriber

**Files:**
- Create: `Sources/MoonshineKit/LiveTranscriber.swift`
- Test: `Tests/MoonshineKitTests/LiveTranscriberTests.swift`

**Interfaces:**
- Consumes: `Segmenter` (A2).
- Produces:
```swift
public protocol Transcribing: AnyObject { func transcribe(_ samples: [Int16]) throws -> String }
public final class LiveTranscriber {
    public var onPartial: ((String) -> Void)?
    public var onFinal: ((String) -> Void)?
    public var onError: ((Error) -> Void)?
    public init(transcriber: Transcribing, segmenter: Segmenter = Segmenter(), queue: DispatchQueue = ...)
    public func feed(_ samples: [Int16])   // any thread
    public func flush()                    // finalise the open segment
    public func reset()                    // drop everything queued, re-arm
    public var isIdle: Bool                // nothing running or queued
}
```

- [ ] **Step 1: Failing tests**

`Tests/MoonshineKitTests/LiveTranscriberTests.swift`:
```swift
import XCTest
@testable import MoonshineKit

/// Records what it was asked to transcribe; can block on a gate to simulate slow inference.
private final class FakeTranscriber: Transcribing {
    private let lock = NSLock()
    private var _calls: [[Int16]] = []
    var calls: [[Int16]] { lock.lock(); defer { lock.unlock() }; return _calls }
    var result = "hello"
    var error: Error?
    /// While set, every call waits on it. Clear it, then signal, to let the blocked call through.
    var gate: DispatchSemaphore?

    func transcribe(_ samples: [Int16]) throws -> String {
        lock.lock()
        _calls.append(samples)
        let gate = self.gate
        let error = self.error
        lock.unlock()
        gate?.wait()
        if let error { throw error }
        return result
    }
    func release() {
        lock.lock(); let g = gate; gate = nil; lock.unlock()
        g?.signal()
    }
}

private struct Boom: Error {}

final class LiveTranscriberTests: XCTestCase {
    private let chunk = 1_600
    private func silence() -> [Int16] { Array(repeating: 0, count: chunk) }
    private func speech() -> [Int16] { Array(repeating: 1_000, count: chunk) }

    private var fake = FakeTranscriber()
    private var partials: [String] = []
    private var finals: [String] = []
    private var errors = 0
    private let lock = NSLock()

    /// A fresh pipeline with 5 chunks of silence already fed, so the pre-roll is full and
    /// the first interim lands on the 6th speech chunk (12_800 samples), as in SegmenterTests.
    private func makeLive() -> LiveTranscriber {
        let live = LiveTranscriber(transcriber: fake)
        live.onPartial = { [self] t in lock.lock(); partials.append(t); lock.unlock() }
        live.onFinal = { [self] t in lock.lock(); finals.append(t); lock.unlock() }
        live.onError = { [self] _ in lock.lock(); errors += 1; lock.unlock() }
        hush(live, chunks: 5)
        return live
    }

    /// Polls until the condition holds or 2 s pass.
    private func waitUntil(_ condition: @escaping () -> Bool, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = Date().addingTimeInterval(2)
        while !condition() && Date() < deadline { Thread.sleep(forTimeInterval: 0.005) }
        XCTAssertTrue(condition(), "timed out", file: file, line: line)
    }
    private func speak(_ live: LiveTranscriber, chunks: Int) { for _ in 0..<chunks { live.feed(speech()) } }
    private func hush(_ live: LiveTranscriber, chunks: Int) { for _ in 0..<chunks { live.feed(silence()) } }

    func testPartialThenFinal() {
        let live = makeLive()
        speak(live, chunks: 6)                 // first interim on the 6th chunk
        waitUntil { self.partials == ["hello"] }
        hush(live, chunks: 7)                  // closes the segment
        waitUntil { self.finals == ["hello"] }
        waitUntil { live.isIdle }
        XCTAssertEqual(fake.calls.map(\.count), [12_800, 12_800 + 7 * 1_600])
    }

    func testLatestInterimWinsWhileBusy() {
        fake.gate = DispatchSemaphore(value: 0)
        let live = makeLive()
        speak(live, chunks: 6)                 // interim 1 (12_800) starts and blocks
        waitUntil { self.fake.calls.count == 1 }
        speak(live, chunks: 16)                // interims at chunk 14 (25_600) and 22 (38_400) queue; only the last survives
        fake.release()
        waitUntil { live.isIdle }
        XCTAssertEqual(fake.calls.map(\.count), [12_800, 38_400])
        XCTAssertEqual(partials, ["hello", "hello"])
    }

    func testStalePartialIsDroppedOnceTheSegmentClosed() {
        fake.gate = DispatchSemaphore(value: 0)
        let live = makeLive()
        speak(live, chunks: 6)
        waitUntil { self.fake.calls.count == 1 }   // interim running, blocked
        hush(live, chunks: 7)                      // segment closes; final queued
        fake.release()
        waitUntil { self.finals == ["hello"] }
        waitUntil { live.isIdle }
        XCTAssertEqual(partials, [], "a partial for a closed segment must not be reported")
    }

    func testFlushFinalisesTheOpenSegment() {
        let live = makeLive()
        speak(live, chunks: 3)
        live.flush()
        waitUntil { self.finals == ["hello"] }
        XCTAssertEqual(fake.calls.map(\.count), [4_800 + 2 * 1_600])
    }

    func testEmptyTextIsNotReported() {
        fake.result = "   "
        let live = makeLive()
        speak(live, chunks: 3)
        live.flush()
        waitUntil { live.isIdle && !self.fake.calls.isEmpty }
        XCTAssertEqual(finals, [])
        XCTAssertEqual(partials, [])
    }

    func testErrorsAreReportedAndDoNotStopTheNextSegment() {
        fake.error = Boom()
        let live = makeLive()
        speak(live, chunks: 3); live.flush()
        waitUntil { self.errors == 1 }
        fake.error = nil
        speak(live, chunks: 3); live.flush()
        waitUntil { self.finals == ["hello"] }
    }

    func testBacklogSkipsInterims() {
        fake.gate = DispatchSemaphore(value: 0)
        let live = makeLive()
        // Two complete utterances while the first job is stuck, then a third still talking.
        speak(live, chunks: 3); hush(live, chunks: 7)
        waitUntil { self.fake.calls.count == 1 }   // final #1 running, blocked
        speak(live, chunks: 3); hush(live, chunks: 7)   // final #2 queued
        speak(live, chunks: 3); hush(live, chunks: 7)   // final #3 queued → two waiting
        speak(live, chunks: 8)                          // pre-roll is empty after a close, so the interim is on the 8th chunk — and skipped: 2 finals wait
        fake.release()
        waitUntil { live.isIdle }
        XCTAssertEqual(fake.calls.count, 3, "three finals, no interim: \(fake.calls.map(\.count))")
        XCTAssertEqual(partials, [])
        XCTAssertEqual(finals, ["hello", "hello", "hello"])
    }

    func testResetDropsQueuedWork() {
        fake.gate = DispatchSemaphore(value: 0)
        let live = makeLive()
        speak(live, chunks: 6)
        waitUntil { self.fake.calls.count == 1 }
        hush(live, chunks: 7)                           // closes segment 0: a final queued behind the stuck interim
        live.reset()
        fake.release()
        waitUntil { live.isIdle }
        XCTAssertEqual(fake.calls.count, 1)
        XCTAssertEqual(finals, [])
    }
}
```
Run: `swift test --filter LiveTranscriberTests` → build failure, `cannot find 'LiveTranscriber'`.

- [ ] **Step 2: Implement**

`Sources/MoonshineKit/LiveTranscriber.swift`:
```swift
import Foundation
import os

/// Anything that turns one utterance of 16 kHz Int16 PCM into text.
public protocol Transcribing: AnyObject {
    func transcribe(_ samples: [Int16]) throws -> String
}

/// Feeds mic audio through a `Segmenter`, runs the transcriber one job at a
/// time off the audio thread, and reports partial and final text.
///
/// Finals are never dropped and go first. Of the interims, only the newest
/// waits — if inference is busy when the next tick arrives, the older audio is
/// superseded — and a partial whose segment has already closed is discarded,
/// so a final is never followed by a stale gray line. If two or more finals are
/// waiting, inference is behind real time and interims are skipped entirely
/// until it catches up.
public final class LiveTranscriber {
    public var onPartial: ((String) -> Void)?
    public var onFinal: ((String) -> Void)?
    public var onError: ((Error) -> Void)?

    /// Finals waiting before interims are skipped.
    static let backlogLimit = 2

    private let transcriber: Transcribing
    private let queue: DispatchQueue
    private let lock = NSLock()
    private let freshSegmenter: Segmenter
    private var segmenter: Segmenter
    private var pendingInterim: (audio: [Int16], segment: Int)?
    private var pendingFinals: [(audio: [Int16], segment: Int)] = []
    private var busy = false
    private var warnedBacklog = false
    private let log = Logger(subsystem: "MoonshineKit", category: "LiveTranscriber")

    public init(transcriber: Transcribing,
                segmenter: Segmenter = Segmenter(),
                queue: DispatchQueue = DispatchQueue(label: "moonshine.inference", qos: .userInitiated)) {
        self.transcriber = transcriber
        self.freshSegmenter = segmenter
        self.segmenter = segmenter
        self.queue = queue
    }

    /// True when nothing is running or waiting.
    public var isIdle: Bool {
        lock.lock(); defer { lock.unlock() }
        return !busy && pendingInterim == nil && pendingFinals.isEmpty
    }

    /// Safe from the audio thread: segments under a lock, then returns.
    public func feed(_ samples: [Int16]) {
        lock.lock()
        enqueue(segmenter.feed(samples))
        let start = startIfNeeded()
        lock.unlock()
        if start { queue.async { [weak self] in self?.drain() } }
    }

    /// Ends the open segment and transcribes it as a final.
    public func flush() {
        lock.lock()
        if let event = segmenter.flush() { enqueue([event]) }
        let start = startIfNeeded()
        lock.unlock()
        if start { queue.async { [weak self] in self?.drain() } }
    }

    /// Drops every queued job and re-arms the segmenter. A job already running
    /// finishes, but its result is discarded because its segment is gone.
    public func reset() {
        lock.lock()
        segmenter = freshSegmenter
        pendingInterim = nil
        pendingFinals = []
        warnedBacklog = false
        generation += 1
        lock.unlock()
    }

    // MARK: - Under the lock

    private func enqueue(_ events: [Segmenter.Event]) {
        for event in events {
            switch event {
            case .interim(let audio, let segment):
                if pendingFinals.count >= Self.backlogLimit {
                    if !warnedBacklog {
                        warnedBacklog = true
                        log.warning("inference is behind real time; skipping interim transcriptions")
                    }
                    pendingInterim = nil
                } else {
                    pendingInterim = (audio, segment)
                }
            case .final(let audio, let segment):
                if pendingInterim?.segment == segment { pendingInterim = nil }
                pendingFinals.append((audio, segment))
            }
        }
    }

    private func startIfNeeded() -> Bool {
        guard !busy, pendingInterim != nil || !pendingFinals.isEmpty else { return false }
        busy = true
        return true
    }

    /// Bumped by reset(); a job carries the generation it was queued in, and
    /// results from an older generation are dropped.
    private var generation = 0

    private struct Job {
        let audio: [Int16]
        let segment: Int
        let isFinal: Bool
        let generation: Int
    }

    private func nextJob() -> Job? {
        lock.lock(); defer { lock.unlock() }
        if !pendingFinals.isEmpty {
            let f = pendingFinals.removeFirst()
            return Job(audio: f.audio, segment: f.segment, isFinal: true, generation: generation)
        }
        if let i = pendingInterim {
            pendingInterim = nil
            return Job(audio: i.audio, segment: i.segment, isFinal: false, generation: generation)
        }
        busy = false
        return nil
    }

    private func segmentIsOpen(_ segment: Int, generation: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return self.generation == generation && segmenter.isOpen && segmenter.currentSegment == segment
    }

    private func generationMatches(_ generation: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return self.generation == generation
    }

    // MARK: - Inference thread

    private func drain() {
        while let job = nextJob() {
            if !job.isFinal, !segmentIsOpen(job.segment, generation: job.generation) { continue }
            do {
                let text = try transcriber.transcribe(job.audio)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty, generationMatches(job.generation) else { continue }
                if job.isFinal {
                    onFinal?(text)
                } else if segmentIsOpen(job.segment, generation: job.generation) {
                    onPartial?(text)
                }
            } catch {
                if generationMatches(job.generation) { onError?(error) }
            }
        }
    }
}
```
- [ ] **Step 3: Run tests**

Run: `swift test --filter LiveTranscriberTests`
Expected: `Executed 8 tests, with 0 failures`. Flaky timing is a bug, not bad luck: every wait is on a condition, never a fixed sleep.

- [ ] **Step 4: Commit**

```bash
git add Sources/MoonshineKit/LiveTranscriber.swift Tests/MoonshineKitTests/LiveTranscriberTests.swift
git commit -m "feat: LiveTranscriber runs segments serially with coalesced interims"
```

---

### Task A4: Python conversion — vocab, encoder, decoder, compiled models

**Files:**
- Create: `convert/pyproject.toml`, `convert/export_vocab.py`, `convert/convert.py`, `convert/README.md`

**Interfaces:**
- Produces: `build/Encoder.mlpackage`, `build/Decoder.mlpackage`, `build/Encoder.mlmodelc`, `build/Decoder.mlmodelc`, `build/vocab.json`.
- Encoder: input `audio` float32 `[1, T]`, T ∈ {16000·s | s = 1…12}; output `encoder_states` float16 `[1, F, 288]`.
- Decoder: inputs `token` int32 `[1,1]`, `encoder_states` float16 `[1,500,288]`, `frames` int32 `[1]`, `position` int32 `[1]`; states `k_cache`, `v_cache` float16 `[6,1,8,194,36]`; output `next_token` int32 `[1]`.

- [ ] **Step 1: Environment**

`convert/pyproject.toml`:
```toml
[project]
name = "moonshine-coreml-convert"
version = "0.1.0"
description = "Converts UsefulSensors/moonshine-tiny to Core ML"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.5,<2.9",
    "transformers>=4.56,<5",
    "coremltools>=8.3",
    "numpy",
    "soundfile",
    "huggingface_hub",
]

[tool.uv]
package = false
```
Run: `cd convert && uv sync && uv run python -c "import torch, transformers, coremltools as ct; print(torch.__version__, transformers.__version__, ct.__version__)"`
Expected: three versions print. If `coremltools` warns that the installed torch is newer than it was tested with, pin `torch` in `pyproject.toml` to the highest version the warning names and `uv sync` again — a torch/coremltools mismatch is the usual cause of opaque trace-conversion failures.

- [ ] **Step 2: `export_vocab.py`**

```python
"""Writes vocab.json: a JSON array of 32768 strings, index = token id."""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "UsefulSensors/moonshine-tiny"
VOCAB_SIZE = 32768

def main(out: Path) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab = [""] * VOCAB_SIZE
    for token, idx in tok.get_vocab().items():
        if idx < VOCAB_SIZE:
            vocab[idx] = token
    assert vocab[1] == "<s>" and vocab[2] == "</s>", (vocab[:3])
    assert vocab[32000].startswith("<<ST_"), vocab[32000]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False))
    filled = sum(1 for v in vocab if v)
    print(f"wrote {out} ({filled} of {VOCAB_SIZE} ids named)")

if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "../build/vocab.json"))
```
Run: `cd convert && uv run python export_vocab.py ../build/vocab.json`
Expected: `wrote ../build/vocab.json (32535 of 32768 ids named)` (a count near that; the tail ids are unused). `python3 -c "import json;v=json.load(open('build/vocab.json'));print(len(v), v[1], v[2], [t for t in v if t.startswith('<0x')][:2])"` → `32768 <s> </s> ['<0x00>', '<0x01>']`.

- [ ] **Step 3: `convert.py`**

```python
"""Converts UsefulSensors/moonshine-tiny to two Core ML models.

Encoder: enumerated audio lengths 1..12 s → encoder_states [1, F, 288] fp16.
Decoder: one token per call, fixed shapes, self-attention KV cache in Core ML
state, argmax inside. See docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md.
"""
import argparse, shutil, subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import MoonshineForConditionalGeneration
from transformers.models.moonshine.modeling_moonshine import apply_rotary_pos_emb

MODEL_ID = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000
BUCKET_SECONDS = list(range(1, 13))
MAX_FRAMES = 500          # 12 s of audio is 498 encoder frames
MAX_POSITIONS = 194       # bos + 193 tokens (config.max_position_embeddings)
NEG = -1e4                # additive mask value; fp16-safe


class Encoder(torch.nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.enc = hf.model.encoder

    def forward(self, audio):                      # [1, T] float32 in [-1, 1]
        return self.enc(audio).last_hidden_state   # [1, F, 288]


class Decoder(torch.nn.Module):
    """One greedy step. Re-implements the layer math with the HF submodules so
    the trace has no Cache objects and no dynamic slices: the causal mask, the
    encoder padding mask and the one-hot cache write all come from `position`
    and `frames`."""

    def __init__(self, hf):
        super().__init__()
        cfg = hf.config
        self.dec = hf.model.decoder
        self.proj_out = hf.proj_out
        self.heads = cfg.decoder_num_attention_heads
        self.head_dim = cfg.hidden_size // self.heads          # 36; HF pads to 40 for its kernel only
        self.hidden = cfg.hidden_size
        n_layers = len(self.dec.layers)
        shape = (n_layers, 1, self.heads, MAX_POSITIONS, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape))
        self.register_buffer("v_cache", torch.zeros(shape))

    def forward(self, token, encoder_states, frames, position):
        # token [1,1] int32 · encoder_states [1,MAX_FRAMES,288] · frames [1] int32 · position [1] int32
        position = position.long()
        frames = frames.long()
        x = self.dec.embed_tokens(token.long())                                # [1,1,288]
        cos, sin = self.dec.rotary_emb(x, position.view(1, 1))                 # rotary for this one position
        pos = position.view(1, 1, 1, 1)
        tok_idx = torch.arange(MAX_POSITIONS).view(1, 1, 1, MAX_POSITIONS)
        causal = (tok_idx > pos).to(x.dtype) * NEG                             # [1,1,1,P]
        write = (tok_idx == pos).to(x.dtype).view(1, 1, MAX_POSITIONS, 1)      # [1,1,P,1]
        frm_idx = torch.arange(MAX_FRAMES).view(1, 1, 1, MAX_FRAMES)
        enc_mask = (frm_idx >= frames.view(1, 1, 1, 1)).to(x.dtype) * NEG      # [1,1,1,F]

        for i, layer in enumerate(self.dec.layers):
            h = layer.input_layernorm(x)
            x = x + self._self_attn(layer.self_attn, i, h, cos, sin, causal, write)
            h = layer.post_attention_layernorm(x)
            x = x + self._cross_attn(layer.encoder_attn, h, encoder_states, enc_mask)
            h = layer.final_layernorm(x)
            x = x + layer.mlp(h)
        x = self.dec.norm(x)
        logits = self.proj_out(x)                                              # [1,1,32768]
        return torch.argmax(logits, dim=-1).to(torch.int32).view(1)

    def _heads(self, t, n):
        return t.view(1, n, self.heads, self.head_dim).transpose(1, 2)         # [1,H,n,D]

    def _self_attn(self, attn, i, h, cos, sin, causal, write):
        q = self._heads(attn.q_proj(h), 1)
        k = self._heads(attn.k_proj(h), 1)
        v = self._heads(attn.v_proj(h), 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        keep = 1.0 - write
        self.k_cache[i : i + 1] = (self.k_cache[i : i + 1] * keep + k * write)
        self.v_cache[i : i + 1] = (self.v_cache[i : i + 1] * keep + v * write)
        K = self.k_cache[i]                                                    # [1,H,P,D]
        V = self.v_cache[i]
        scores = torch.matmul(q, K.transpose(-1, -2)) * attn.scaling + causal  # [1,H,1,P]
        w = torch.softmax(scores, dim=-1)
        o = torch.matmul(w, V).transpose(1, 2).reshape(1, 1, self.heads * self.head_dim)
        return attn.o_proj(o)

    def _cross_attn(self, attn, h, enc, mask):
        q = self._heads(attn.q_proj(h), 1)
        k = self._heads(attn.k_proj(enc), MAX_FRAMES)
        v = self._heads(attn.v_proj(enc), MAX_FRAMES)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling + mask    # [1,H,1,F]
        w = torch.softmax(scores, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).reshape(1, 1, self.heads * self.head_dim)
        return attn.o_proj(o)


def convert_encoder(hf, out: Path) -> Path:
    enc = Encoder(hf).eval()
    example = torch.zeros(1, 4 * SAMPLE_RATE)
    with torch.no_grad():
        traced = torch.jit.trace(enc, example)
    shapes = ct.EnumeratedShapes(shapes=[[1, s * SAMPLE_RATE] for s in BUCKET_SECONDS],
                                 default=[1, 4 * SAMPLE_RATE])
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=shapes, dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_states", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    model.short_description = "Moonshine Tiny encoder (UsefulSensors/moonshine-tiny), MIT"
    path = out / "Encoder.mlpackage"
    model.save(str(path))
    return path


def convert_decoder(hf, out: Path) -> Path:
    dec = Decoder(hf).eval()
    example = (torch.ones(1, 1, dtype=torch.int32),
               torch.zeros(1, MAX_FRAMES, hf.config.hidden_size),
               torch.tensor([40], dtype=torch.int32),
               torch.tensor([0], dtype=torch.int32))
    with torch.no_grad():
        traced = torch.jit.trace(dec, example)
    cache_shape = tuple(dec.k_cache.shape)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="encoder_states", shape=(1, MAX_FRAMES, hf.config.hidden_size), dtype=np.float16),
            ct.TensorType(name="frames", shape=(1,), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="next_token", dtype=np.int32)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="k_cache"),
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="v_cache"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    model.short_description = "Moonshine Tiny decoder step (UsefulSensors/moonshine-tiny), MIT"
    path = out / "Decoder.mlpackage"
    model.save(str(path))
    return path


def compile_model(package: Path, out: Path) -> Path:
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)], check=True)
    compiled = out / (package.stem + ".mlmodelc")
    assert compiled.is_dir(), compiled
    return compiled


def smoke_test(out: Path) -> None:
    enc = ct.models.MLModel(str(out / "Encoder.mlpackage"))
    dec = ct.models.MLModel(str(out / "Decoder.mlpackage"))
    states = enc.predict({"audio": np.zeros((1, SAMPLE_RATE), dtype=np.float32)})["encoder_states"]
    assert states.shape == (1, 40, 288), states.shape   # 1 s → 40 frames
    padded = np.zeros((1, MAX_FRAMES, 288), dtype=np.float16)
    padded[:, :40] = states
    state = dec.make_state()
    out_token = dec.predict({"token": np.array([[1]], dtype=np.int32), "encoder_states": padded,
                             "frames": np.array([40], dtype=np.int32),
                             "position": np.array([0], dtype=np.int32)}, state=state)["next_token"]
    assert out_token.shape == (1,) and out_token.dtype == np.int32, (out_token.shape, out_token.dtype)
    print(f"smoke test ok: encoder {states.shape}, first decoder token {int(out_token[0])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../build"))
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hf = MoonshineForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    if not args.skip_encoder:
        compile_model(convert_encoder(hf, args.out), args.out)
    if not args.skip_decoder:
        compile_model(convert_decoder(hf, args.out), args.out)
    if not (args.out / "vocab.json").exists():
        subprocess.run(["uv", "run", "python", "export_vocab.py", str(args.out / "vocab.json")], check=True)
    smoke_test(args.out)
    print("models in", args.out.resolve())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the conversion**

Run: `cd convert && uv run python convert.py --out ../build`
Expected: two `.mlpackage` and two `.mlmodelc` in `build/`, then `smoke test ok: encoder (1, 40, 288), first decoder token N`. `du -sh build/*.mlmodelc` → encoder ≈ 16 MB, decoder ≈ 38 MB (fp16; the decoder carries the 32768×288 embedding twice — `embed_tokens` and `proj_out` are tied in HF but traced as two constants; acceptable for v1).

If `torch.jit.trace` warns about in-place modification of buffers, that is expected (it is the stateful pattern). If `ct.convert` fails on the decoder, the three likeliest causes, in order: (1) the in-place slice assignment was not recognised — make sure the assignment is exactly `self.k_cache[i : i + 1] = ...` on the registered buffer; (2) the `.long()` casts on int32 inputs — try removing them and comparing against int32 `arange`s instead; (3) `apply_rotary_pos_emb` signature changed in the installed transformers — open its source and match the call. If the *encoder* fails on enumerated shapes, retry with `ct.RangeDim(lower_bound=SAMPLE_RATE, upper_bound=12 * SAMPLE_RATE, default=4 * SAMPLE_RATE)` in place of `EnumeratedShapes` and note it in `convert/README.md` (performance on the Neural Engine is then to be measured in A7).

- [ ] **Step 5: `convert/README.md`**

Document: `uv sync`, `uv run python convert.py`, what lands in `build/`, the input/output contract (copy the Interfaces block above), and that the smoke test proves shapes only — `parity_test.py` (A5) proves correctness.

- [ ] **Step 6: Commit**

```bash
git add convert/pyproject.toml convert/uv.lock convert/export_vocab.py convert/convert.py convert/README.md
git commit -m "feat(convert): Core ML export of the Moonshine Tiny encoder and stateful decoder"
```
(`build/` is gitignored.)

---

### Task A5: Parity test and golden transcripts

**Files:**
- Create: `convert/parity_test.py`
- Create: `test-assets/<name>.txt` (golden, written by the script)

**Interfaces:**
- Consumes: `build/Encoder.mlpackage`, `build/Decoder.mlpackage` (A4), `test-assets/*.wav` (A1).
- Produces: `test-assets/<name>.txt` — the HF greedy transcript, used by the Swift integration test (A6).

- [ ] **Step 1: Write `parity_test.py`**

```python
"""HF (PyTorch fp32) vs Core ML (fp16) greedy transcripts on test-assets/*.wav.

PASS when the decoded texts match for every clip. Also prints the token ids so
an fp16 near-tie that changes a token without changing the text is visible.
--write-golden saves the HF text as test-assets/<name>.txt for the Swift tests.
"""
import argparse, sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

MODEL_ID = "UsefulSensors/moonshine-tiny"
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
    states = enc.predict({"audio": padded_audio})["encoder_states"]          # [1, F, 288]
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
    ap.add_argument("--models", type=Path, default=Path("../build"))
    ap.add_argument("--assets", type=Path, default=Path("../test-assets"))
    ap.add_argument("--write-golden", action="store_true")
    args = ap.parse_args()

    model = MoonshineForConditionalGeneration.from_pretrained(MODEL_ID).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
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
```

- [ ] **Step 2: Run it**

Run: `cd convert && uv run python parity_test.py --write-golden`
Expected: `PASS` for each of `coffee.wav`, `hello.wav`, `weather.wav`, `all clips match`, and three `test-assets/*.txt` written. Eyeball the texts against `test-assets/*.say.txt` — they should read as the spoken sentences (punctuation/casing may differ; that is Moonshine's choice, not a bug).

If a clip FAILs with texts that differ by one token near the end, first check `max_new_tokens` is identical on both sides (it is computed from the same `len(audio)`); then re-run the decoder conversion with `compute_precision=ct.precision.FLOAT32` as a diagnostic — if fp32 passes and fp16 fails, the fp16 near-tie is real: keep fp16, and change the PASS criterion to "texts equal after lowercasing and stripping punctuation" with a comment explaining why. If fp32 also fails, the wrapper math is wrong — compare the decoder step against HF layer by layer (`hf.model.decoder(...)` with `use_cache=False` on the same prefix) before touching anything else.

- [ ] **Step 3: Commit**

```bash
git add convert/parity_test.py test-assets/*.txt
git commit -m "test(convert): HF vs Core ML parity check and golden transcripts"
```

---

### Task A6: MoonshineModel, Transcriber, WAVReader, integration test

**Files:**
- Create: `Sources/MoonshineKit/MoonshineModel.swift`, `Sources/MoonshineKit/Transcriber.swift`, `Sources/MoonshineKit/WAVReader.swift`
- Test: `Tests/MoonshineKitTests/WAVReaderTests.swift`, `Tests/MoonshineKitTests/TranscriberIntegrationTests.swift`

**Interfaces:**
- Consumes: `TokenDecoder` (A1), `Transcribing` (A3), compiled models in `build/` (A4), golden texts (A5).
- Produces:
```swift
public enum MoonshineError: Error { case missingOutput(String), unsupportedWAV(String) }
public final class MoonshineModel {
    public static let sampleRate = 16_000, maxFrames = 500, hiddenSize = 288, maxPositions = 194
    public static let bucketSeconds = Array(1...12)
    public struct Encoded { public let states: MLMultiArray; public let frames: Int }
    public let vocab: [String]
    public init(directory: URL, computeUnits: MLComputeUnits = .all) throws
    public func encode(_ audio: [Float]) throws -> Encoded
    public func makeDecoderState() -> MLState
    public func decodeStep(token: Int32, encoded: Encoded, position: Int, state: MLState) throws -> Int32
}
public final class Transcriber: Transcribing {
    public init(model: MoonshineModel)
    public func transcribe(_ samples: [Int16]) throws -> String
    public static func maxTokens(forSamples count: Int) -> Int
}
public enum WAVReader { public static func readInt16Mono16k(_ url: URL) throws -> [Int16] }
```

- [ ] **Step 1: Failing WAVReader test**

`Tests/MoonshineKitTests/WAVReaderTests.swift`:
```swift
import XCTest
@testable import MoonshineKit

final class WAVReaderTests: XCTestCase {
    static let assets = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("test-assets")

    func testReadsTheHelloClip() throws {
        let samples = try WAVReader.readInt16Mono16k(Self.assets.appendingPathComponent("hello.wav"))
        XCTAssertGreaterThan(samples.count, 16_000 * 2, "clip is longer than 2 s")
        XCTAssertLessThan(samples.count, 16_000 * 10)
        XCTAssertTrue(samples.contains { abs(Int($0)) > 1_000 }, "not silent")
    }

    func testRejectsNonWAV() {
        let url = Self.assets.appendingPathComponent("hello.txt")
        XCTAssertThrowsError(try WAVReader.readInt16Mono16k(url))
    }
}
```
Run: `swift test --filter WAVReaderTests` → `cannot find 'WAVReader'`.

- [ ] **Step 2: WAVReader**

`Sources/MoonshineKit/WAVReader.swift`:
```swift
import Foundation

public enum MoonshineError: Error, CustomStringConvertible {
    case missingOutput(String)
    case unsupportedWAV(String)
    public var description: String {
        switch self {
        case .missingOutput(let name): return "Core ML output '\(name)' missing"
        case .unsupportedWAV(let why): return "unsupported WAV: \(why)"
        }
    }
}

/// Minimal RIFF reader for the one format MoonshineKit speaks: 16-bit PCM, mono, 16 kHz.
public enum WAVReader {
    public static func readInt16Mono16k(_ url: URL) throws -> [Int16] {
        let data = try Data(contentsOf: url)
        guard data.count > 12, data[0..<4] == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) else {
            throw MoonshineError.unsupportedWAV("not a RIFF/WAVE file")
        }
        var offset = 12
        var format: (channels: Int, rate: Int, bits: Int, pcm: Bool)?
        while offset + 8 <= data.count {
            let id = String(decoding: data[offset..<offset + 4], as: UTF8.self)
            let size = Int(data.readUInt32LE(at: offset + 4))
            let body = offset + 8
            if id == "fmt " {
                guard body + 16 <= data.count else { throw MoonshineError.unsupportedWAV("short fmt chunk") }
                format = (channels: Int(data.readUInt16LE(at: body + 2)),
                          rate: Int(data.readUInt32LE(at: body + 4)),
                          bits: Int(data.readUInt16LE(at: body + 14)),
                          pcm: data.readUInt16LE(at: body) == 1)
            } else if id == "data" {
                guard let f = format else { throw MoonshineError.unsupportedWAV("data before fmt") }
                guard f.pcm, f.bits == 16, f.channels == 1, f.rate == 16_000 else {
                    throw MoonshineError.unsupportedWAV("need 16-bit PCM mono 16 kHz, got \(f)")
                }
                let end = min(body + size, data.count)
                let count = (end - body) / 2
                var samples = [Int16](repeating: 0, count: count)
                _ = samples.withUnsafeMutableBytes { data.copyBytes(to: $0, from: body..<(body + count * 2)) }
                return samples
            }
            offset = body + size + (size & 1)
        }
        throw MoonshineError.unsupportedWAV("no data chunk")
    }
}

private extension Data {
    func readUInt16LE(at i: Int) -> UInt16 { UInt16(self[i]) | UInt16(self[i + 1]) << 8 }
    func readUInt32LE(at i: Int) -> UInt32 {
        UInt32(self[i]) | UInt32(self[i + 1]) << 8 | UInt32(self[i + 2]) << 16 | UInt32(self[i + 3]) << 24
    }
}
```
Run: `swift test --filter WAVReaderTests` → 2 tests pass. (Indexing `Data` with absolute offsets is correct here because `Data(contentsOf:)` starts at index 0.)

- [ ] **Step 3: Failing integration test**

`Tests/MoonshineKitTests/TranscriberIntegrationTests.swift`:
```swift
import XCTest
@testable import MoonshineKit

/// Runs the real Core ML models. Skipped unless MOONSHINE_MODELS points at a
/// directory holding Encoder.mlmodelc, Decoder.mlmodelc and vocab.json:
///   MOONSHINE_MODELS=$PWD/build swift test --filter TranscriberIntegrationTests
final class TranscriberIntegrationTests: XCTestCase {
    static let assets = WAVReaderTests.assets

    private func loadModel() throws -> MoonshineModel {
        guard let dir = ProcessInfo.processInfo.environment["MOONSHINE_MODELS"] else {
            throw XCTSkip("set MOONSHINE_MODELS to run")
        }
        return try MoonshineModel(directory: URL(fileURLWithPath: dir))
    }

    /// Lowercase, letters/digits/spaces only, single-spaced — so punctuation choices don't fail the test.
    static func normalize(_ s: String) -> String {
        s.lowercased().unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .split(separator: " ").joined(separator: " ")
    }

    func testTranscribesEveryClipLikeTheReference() throws {
        let transcriber = Transcriber(model: try loadModel())
        let clips = try FileManager.default.contentsOfDirectory(at: Self.assets, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "wav" }.sorted { $0.path < $1.path }
        XCTAssertEqual(clips.count, 3)
        for wav in clips {
            let expected = try String(contentsOf: wav.deletingPathExtension().appendingPathExtension("txt"), encoding: .utf8)
            let samples = try WAVReader.readInt16Mono16k(wav)
            let text = try transcriber.transcribe(samples)
            XCTAssertEqual(Self.normalize(text), Self.normalize(expected), wav.lastPathComponent)
        }
    }

    func testSilenceIsEmptyOrNearlySo() throws {
        let transcriber = Transcriber(model: try loadModel())
        let text = try transcriber.transcribe([Int16](repeating: 0, count: 16_000))
        XCTAssertLessThan(Self.normalize(text).count, 20, "silence should not hallucinate a sentence: \(text)")
    }

    func testEncoderFramesPerBucket() throws {
        let model = try loadModel()
        XCTAssertEqual(try model.encode([Float](repeating: 0, count: 16_000)).frames, 40)
        XCTAssertEqual(try model.encode([Float](repeating: 0, count: 12 * 16_000)).frames, 498)
        XCTAssertEqual(try model.encode([Float](repeating: 0, count: 20 * 16_000)).frames, 498, "over-long audio is truncated to 12 s")
    }

    func testMaxTokensRule() {
        XCTAssertEqual(Transcriber.maxTokens(forSamples: 16_000), 8)        // 6.5 + 2 → 8
        XCTAssertEqual(Transcriber.maxTokens(forSamples: 12 * 16_000), 80)  // 78 + 2
        XCTAssertEqual(Transcriber.maxTokens(forSamples: 60 * 16_000), 193) // capped
    }
}
```
Run: `MOONSHINE_MODELS=$PWD/build swift test --filter TranscriberIntegrationTests` → `cannot find 'MoonshineModel'`.

- [ ] **Step 4: MoonshineModel**

`Sources/MoonshineKit/MoonshineModel.swift`:
```swift
import CoreML
import Foundation

/// The two compiled Core ML models plus the vocabulary, loaded from one
/// directory — a moonshine-coreml release unzipped: `Encoder.mlmodelc`,
/// `Decoder.mlmodelc`, `vocab.json`. Loading takes a second or two; keep the
/// instance for the life of the app.
public final class MoonshineModel {
    public static let sampleRate = 16_000
    /// Encoder input buckets, in seconds. Audio is zero-padded up to the next
    /// bucket; anything longer than the last is truncated to it.
    public static let bucketSeconds = Array(1...12)
    /// The decoder's fixed encoder-state length; 12 s of audio is 498 frames.
    public static let maxFrames = 500
    public static let hiddenSize = 288
    /// Decoder positions: bos plus up to 193 generated tokens.
    public static let maxPositions = 194

    public struct Encoded {
        /// `[1, maxFrames, hiddenSize]` float16, zero beyond `frames`.
        public let states: MLMultiArray
        public let frames: Int
    }

    private let encoder: MLModel
    private let decoder: MLModel
    public let vocab: [String]

    public init(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        encoder = try MLModel(contentsOf: directory.appendingPathComponent("Encoder.mlmodelc"), configuration: config)
        decoder = try MLModel(contentsOf: directory.appendingPathComponent("Decoder.mlmodelc"), configuration: config)
        vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
    }

    /// Runs the encoder on `audio` (Float in -1...1) padded to its bucket, and
    /// lays the result into the decoder's fixed-size input.
    public func encode(_ audio: [Float]) throws -> Encoded {
        let seconds = min(Self.bucketSeconds.last!,
                          max(1, Int((Double(audio.count) / Double(Self.sampleRate)).rounded(.up))))
        let length = seconds * Self.sampleRate
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float32)
        let p = input.dataPointer.bindMemory(to: Float.self, capacity: length)
        p.initialize(repeating: 0, count: length)
        let n = min(audio.count, length)
        audio.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: n) }

        let out = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": MLFeatureValue(multiArray: input)]))
        guard let states = out.featureValue(for: "encoder_states")?.multiArrayValue else {
            throw MoonshineError.missingOutput("encoder_states")
        }
        let frames = min(states.shape[1].intValue, Self.maxFrames)
        let padded = try MLMultiArray(shape: [1, NSNumber(value: Self.maxFrames), NSNumber(value: Self.hiddenSize)], dataType: .float16)
        let total = Self.maxFrames * Self.hiddenSize
        memset(padded.dataPointer, 0, total * MemoryLayout<Float16>.size)
        let used = frames * Self.hiddenSize
        if states.dataType == .float16, Self.isContiguous(states) {
            memcpy(padded.dataPointer, states.dataPointer, used * MemoryLayout<Float16>.size)
        } else {
            for i in 0..<used { padded[i] = states[i] }   // slow path: Core ML handed back another layout
        }
        return Encoded(states: padded, frames: frames)
    }

    public func makeDecoderState() -> MLState { decoder.makeState() }

    /// One greedy step. `position` is the index of `token` in the sequence (0 for bos).
    public func decodeStep(token: Int32, encoded: Encoded, position: Int, state: MLState) throws -> Int32 {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: token)
        let frames = try MLMultiArray(shape: [1], dataType: .int32)
        frames[0] = NSNumber(value: Int32(encoded.frames))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenArray),
            "encoder_states": MLFeatureValue(multiArray: encoded.states),
            "frames": MLFeatureValue(multiArray: frames),
            "position": MLFeatureValue(multiArray: pos),
        ])
        let out = try decoder.prediction(from: input, using: state)
        guard let next = out.featureValue(for: "next_token")?.multiArrayValue else {
            throw MoonshineError.missingOutput("next_token")
        }
        return next[0].int32Value
    }

    private static func isContiguous(_ array: MLMultiArray) -> Bool {
        var expected = 1
        for (shape, stride) in zip(array.shape.reversed(), array.strides.reversed()) {
            if stride.intValue != expected { return false }
            expected *= shape.intValue
        }
        return true
    }
}
```

- [ ] **Step 5: Transcriber**

`Sources/MoonshineKit/Transcriber.swift`:
```swift
import CoreML
import Foundation

/// One utterance in, text out: Int16 → Float, encode, greedy decode until eos
/// or the token cap, then `TokenDecoder`. One request at a time — callers
/// serialise (LiveTranscriber does).
public final class Transcriber: Transcribing {
    private let model: MoonshineModel
    private let decoder: TokenDecoder

    public init(model: MoonshineModel) {
        self.model = model
        self.decoder = TokenDecoder(vocab: model.vocab)
    }

    /// Moonshine's rule of thumb: at most 6.5 tokens per second of audio, plus
    /// a little slack, never more than the 193 positions left after bos.
    public static func maxTokens(forSamples count: Int) -> Int {
        let seconds = Double(count) / Double(MoonshineModel.sampleRate)
        return min(MoonshineModel.maxPositions - 1, Int(seconds * 6.5) + 2)
    }

    public func transcribe(_ samples: [Int16]) throws -> String {
        guard !samples.isEmpty else { return "" }
        let audio = samples.map { Float($0) / 32_768 }
        let encoded = try model.encode(audio)
        let state = model.makeDecoderState()
        var ids: [Int32] = []
        var token = TokenDecoder.bos
        for position in 0..<Self.maxTokens(forSamples: samples.count) {
            let next = try model.decodeStep(token: token, encoded: encoded, position: position, state: state)
            if next == TokenDecoder.eos { break }
            ids.append(next)
            token = next
        }
        return decoder.decode(ids)
    }
}
```

- [ ] **Step 6: Run the integration tests**

Run: `MOONSHINE_MODELS=$PWD/build swift test --filter TranscriberIntegrationTests`
Expected: `Executed 4 tests, with 0 failures`. Also `swift test` (no env) → the three model tests report `skipped`, everything else passes.

If `testTranscribesEveryClipLikeTheReference` fails while A5's parity passed, the bug is on the Swift side: print `ids` from `transcribe` and compare with the `coreml=` ids the parity script printed for that clip. Mismatch at the first token → `encode` padding/copy; later → `position`/`state` handling.

- [ ] **Step 7: Commit**

```bash
git add Sources/MoonshineKit Tests/MoonshineKitTests
git commit -m "feat: MoonshineModel and Transcriber run the Core ML models end to end"
```

---

### Task A7: `moonshine-bench`

**Files:**
- Modify: `Sources/moonshine-bench/main.swift` (replace the placeholder)

**Interfaces:**
- Consumes: `MoonshineModel`, `Transcriber`, `LiveTranscriber`, `WAVReader`.

- [ ] **Step 1: Write the CLI**

```swift
import Foundation
import MoonshineKit

// moonshine-bench --models DIR file.wav [--live] [--cpu]
//   Default: transcribe the file twice (cold, warm) and print text + timings.
//   --live: push the file through LiveTranscriber in 100 ms chunks and print partials/finals.
//   --cpu:  computeUnits = .cpuOnly, to compare against the Neural Engine.

var args = Array(CommandLine.arguments.dropFirst())
func take(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...i + 1); return v
}
func has(_ flag: String) -> Bool {
    guard let i = args.firstIndex(of: flag) else { return false }
    args.remove(at: i); return true
}
guard let modelsDir = take("--models") else {
    FileHandle.standardError.write(Data("usage: moonshine-bench --models DIR file.wav [--live] [--cpu]\n".utf8)); exit(2)
}
let live = has("--live"), cpu = has("--cpu")
guard let file = args.first else { FileHandle.standardError.write(Data("missing wav\n".utf8)); exit(2) }

func ms(_ block: () throws -> Void) rethrows -> Double {
    let t = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
}

do {
    let samples = try WAVReader.readInt16Mono16k(URL(fileURLWithPath: file))
    let seconds = Double(samples.count) / 16_000
    var model: MoonshineModel!
    let load = try ms { model = try MoonshineModel(directory: URL(fileURLWithPath: modelsDir), computeUnits: cpu ? .cpuOnly : .all) }
    print(String(format: "loaded in %.0f ms (%@), clip %.2f s", load, cpu ? "cpu" : "all", seconds))
    let transcriber = Transcriber(model: model)

    if live {
        let liveT = LiveTranscriber(transcriber: transcriber)
        let start = DispatchTime.now().uptimeNanoseconds
        func stamp() -> String { String(format: "%7.0f ms", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
        liveT.onPartial = { print("\(stamp())  partial: \($0)") }
        liveT.onFinal = { print("\(stamp())  FINAL:   \($0)") }
        liveT.onError = { print("\(stamp())  error:   \($0)") }
        var i = 0
        while i < samples.count {
            liveT.feed(Array(samples[i..<min(i + 1_600, samples.count)]))
            i += 1_600
            Thread.sleep(forTimeInterval: 0.1)   // real time, so the timings mean something
        }
        liveT.flush()
        while !liveT.isIdle { Thread.sleep(forTimeInterval: 0.01) }
        print("\(stamp())  done")
    } else {
        var text = ""
        let cold = try ms { text = try transcriber.transcribe(samples) }
        let warm = try ms { text = try transcriber.transcribe(samples) }
        print("text: \(text)")
        print(String(format: "cold %.0f ms, warm %.0f ms, RTF %.2f (warm)", cold, warm, warm / 1000 / seconds))
        // Break the warm run down.
        let audio = samples.map { Float($0) / 32_768 }
        var encoded: MoonshineModel.Encoded!
        let enc = try ms { encoded = try model.encode(audio) }
        let state = model.makeDecoderState()
        var token: Int32 = 1, steps = 0
        let dec = try ms {
            for p in 0..<Transcriber.maxTokens(forSamples: samples.count) {
                token = try model.decodeStep(token: token, encoded: encoded, position: p, state: state)
                steps += 1
                if token == 2 { break }
            }
        }
        print(String(format: "encode %.0f ms, decode %.0f ms for %d steps (%.1f ms/token)", enc, dec, steps, dec / Double(max(steps, 1))))
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8)); exit(1)
}
```

- [ ] **Step 2: Run it**

Run: `swift run -c release moonshine-bench --models build test-assets/hello.wav`
Expected: the golden sentence for `hello` and timing lines. Then `--live`: partials appear as the clip plays, one `FINAL:` at the end (or at a pause), `done`. Then `--cpu` once, to see the spread. Record the numbers (M4 Pro, warm) — they go in the README in A8.

- [ ] **Step 3: Commit**

```bash
git add Sources/moonshine-bench/main.swift
git commit -m "feat: moonshine-bench prints transcript and timings, with a --live mode"
```

---

### Task A8: Release packaging, docs, GitHub repo and v0.1.0 release

**Files:**
- Create: `Scripts/package-models.sh`, `Scripts/fetch-models.sh`
- Modify: `README.md`

- [ ] **Step 1: Scripts**

`Scripts/package-models.sh`:
```bash
#!/bin/bash
# Zips the compiled models for a GitHub release: Scripts/package-models.sh 0.1.0
set -euo pipefail
VERSION="${1:?version, e.g. 0.1.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/moonshine-tiny-coreml-v$VERSION.zip"
rm -f "$OUT"
( cd "$ROOT/build" && zip -qr "$OUT" Encoder.mlmodelc Decoder.mlmodelc vocab.json )
ls -lh "$OUT"
```

`Scripts/fetch-models.sh`:
```bash
#!/bin/bash
# Downloads a release's compiled models: Scripts/fetch-models.sh 0.1.0 [DEST]
set -euo pipefail
VERSION="${1:?version, e.g. 0.1.0}"
DEST="${2:-$(cd "$(dirname "$0")/.." && pwd)/build}"
URL="https://github.com/jonyen/moonshine-coreml/releases/download/v$VERSION/moonshine-tiny-coreml-v$VERSION.zip"
TMP="$(mktemp -d)"
curl -fL "$URL" -o "$TMP/models.zip"
mkdir -p "$DEST"
unzip -oq "$TMP/models.zip" -d "$DEST"
rm -rf "$TMP"
echo "models in $DEST:"; ls "$DEST"
```
`chmod +x Scripts/*.sh`.

- [ ] **Step 2: README**

Sections: What this is (2 sentences + link to spec); Models (release asset name, contents, fetch script, "not in git"); Using MoonshineKit (SPM snippet `.package(url: "https://github.com/jonyen/moonshine-coreml", from: "0.1.0")`, the 10-line example: `MoonshineModel(directory:)` → `Transcriber` → `LiveTranscriber` with `onPartial/onFinal`, feed Int16 16 kHz mono); Bench (command + the A7 numbers, labelled "M4 Pro, warm"); Converting yourself (`convert/README.md` pointer, parity test command); Tests (`swift test`, `MOONSHINE_MODELS=$PWD/build swift test`); Related (apple-watch-captions, caption-core, mac-live-captions); License MIT, model MIT (UsefulSensors).

- [ ] **Step 3: Package, create the repo, release**

```bash
Scripts/package-models.sh 0.1.0
git add -A && git commit -m "docs: README, model packaging and fetch scripts"
gh repo create jonyen/moonshine-coreml --public --source . --description "Moonshine Tiny speech-to-text on Core ML for watchOS/iOS/macOS, with the MoonshineKit Swift package" --push
git tag 0.1.0 && git push --tags
gh release create v0.1.0 dist/moonshine-tiny-coreml-v0.1.0.zip --title "v0.1.0 — Moonshine Tiny Core ML models" --notes "Encoder.mlmodelc, Decoder.mlmodelc and vocab.json converted from UsefulSensors/moonshine-tiny (MIT). Unzip into a directory and hand it to MoonshineModel(directory:)."
```
Note the two tags: SPM resolves `from: 0.1.0` against the plain `0.1.0` tag; the release is `v0.1.0`, which the fetch script expects.

- [ ] **Step 4: Verify the round trip**

```bash
rm -rf /tmp/mk && Scripts/fetch-models.sh 0.1.0 /tmp/mk && MOONSHINE_MODELS=/tmp/mk swift test --filter TranscriberIntegrationTests
```
Expected: download, then `Executed 4 tests, with 0 failures`. Part B can now start.

---

## Self-review notes

- Spec coverage: conversion (A4), parity (A5), TokenDecoder (A1), Segmenter (A2), LiveTranscriber (A3), MoonshineModel/Transcriber (A6), bench (A7), release + fetch (A8). Cross-attention precompute and int8 are deliberately absent (spec: later, only if profiling says so).
- Names used across tasks: `Transcribing`, `LiveTranscriber.feed/flush/reset/isIdle`, `MoonshineModel.Encoded`, `decodeStep(token:encoded:position:state:)`, `Transcriber.maxTokens(forSamples:)`, `WAVReader.readInt16Mono16k`, `MoonshineError` — consistent between definition and use.
