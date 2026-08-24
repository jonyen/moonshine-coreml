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
