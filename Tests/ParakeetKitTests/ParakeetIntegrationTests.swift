import XCTest
import MoonshineKit
@testable import ParakeetKit

/// Runs the real Core ML models. Skipped unless PARAKEET_MODELS points at a
/// directory holding Encoder.mlmodelc, CTCHead.mlmodelc and vocab.json:
///   PARAKEET_MODELS=$PWD/build-parakeet swift test --filter ParakeetIntegrationTests
final class ParakeetIntegrationTests: XCTestCase {
    static let assets = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("test-assets")

    private func loadModel() throws -> ParakeetModel {
        guard let dir = ProcessInfo.processInfo.environment["PARAKEET_MODELS"] else {
            throw XCTSkip("set PARAKEET_MODELS to run")
        }
        return try ParakeetModel(directory: URL(fileURLWithPath: dir))
    }

    /// Lowercase, letters/digits/spaces only, single-spaced — Parakeet's
    /// punctuation happens to match the goldens today, but the contract we
    /// hold it to is the words.
    static func normalize(_ s: String) -> String {
        s.lowercased().unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .split(separator: " ").joined(separator: " ")
    }

    func testTranscribesEveryClipLikeTheReference() throws {
        let transcriber = ParakeetTranscriber(model: try loadModel())
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
        let transcriber = ParakeetTranscriber(model: try loadModel())
        let text = try transcriber.transcribe([Int16](repeating: 0, count: 16_000))
        XCTAssertLessThan(Self.normalize(text).count, 20, "silence should not hallucinate a sentence: \(text)")
    }

    func testEncoderLengthTracksAudioLength() throws {
        let model = try loadModel()
        // 3 s → 38 frames at 80 ms/frame (after 8x subsampling), like the spike measured.
        XCTAssertEqual(try model.frameIDs([Float](repeating: 0, count: 3 * 16_000)).count, 38)
        // The full 15 s window fills all 188 frames; longer audio is truncated to it.
        XCTAssertEqual(try model.frameIDs([Float](repeating: 0, count: 15 * 16_000)).count, 188)
        XCTAssertEqual(try model.frameIDs([Float](repeating: 0, count: 20 * 16_000)).count, 188)
    }

    func testWorksThroughLiveTranscriberPipeline() throws {
        // The point of conforming to Transcribing: the existing pipeline reuses
        // unchanged. Feed mic-sized 100 ms chunks so the Segmenter opens on the
        // speech onset instead of swallowing the clip into its pre-roll.
        let transcriber = ParakeetTranscriber(model: try loadModel())
        let samples = try WAVReader.readInt16Mono16k(Self.assets.appendingPathComponent("hello.wav"))
        let live = LiveTranscriber(transcriber: transcriber)
        let lock = NSLock()
        var finals: [String] = []
        live.onFinal = { text in lock.lock(); finals.append(text); lock.unlock() }
        var i = 0
        while i < samples.count {
            live.feed(Array(samples[i..<min(i + 1_600, samples.count)]))
            i += 1_600
        }
        live.flush()
        let deadline = Date().addingTimeInterval(30)
        while live.isIdle == false && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
        lock.lock(); let joined = finals.joined(separator: " "); lock.unlock()
        XCTAssertEqual(Self.normalize(joined),
                       Self.normalize("Hello, this is a test of captions running on the watch."))
    }
}
