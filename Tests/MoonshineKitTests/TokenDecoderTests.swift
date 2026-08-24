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
