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
