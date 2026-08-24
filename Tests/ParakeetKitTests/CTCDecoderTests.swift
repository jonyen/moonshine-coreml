import XCTest
@testable import ParakeetKit

/// Pure-function tests for the greedy CTC decode, with fixtures captured from
/// the Python reference (`convert/parakeet/run_ctc.py`): the actual per-frame
/// argmax ids the Core ML models produce for two test clips, decoded against
/// a sparse vocabulary holding just the pieces those ids touch.
final class CTCDecoderTests: XCTestCase {
    /// A 1024-entry vocab with only the given indices populated.
    private func sparseVocab(_ pieces: [Int: String]) -> [String] {
        var vocab = [String](repeating: "", count: 1024)
        for (i, s) in pieces { vocab[i] = s }
        return vocab
    }

    func testHelloFixtureFromPythonReference() {
        // hello.wav → 38 encoder frames (fp16 build, CPU_ONLY).
        let ids = [1024, 1024, 285, 1024, 30, 969, 1024, 1024, 1024, 988, 1024, 90, 59,
                   1024, 3, 1, 1024, 163, 35, 1024, 903, 1024, 968, 1024, 330, 1024,
                   822, 1024, 541, 62, 1024, 6, 1024, 757, 1024, 1024, 1024, 986]
        let vocab = sparseVocab([1: "▁t", 3: "▁a", 6: "▁the", 30: "ll", 35: "▁of",
                                 59: "▁is", 62: "▁on", 90: "▁this", 163: "est",
                                 285: "▁He", 330: "ions", 541: "ning", 757: "▁watch",
                                 822: "▁run", 903: "▁cap", 968: "t", 969: "o",
                                 986: ".", 988: ","])
        XCTAssertEqual(CTCDecoder(vocab: vocab).decode(ids),
                       "Hello, this is a test of captions running on the watch.")
    }

    func testCoffeeFixtureCollapsesRepeats() {
        // coffee.wav → 31 frames; several ids repeat across adjacent frames
        // (26 26, 189 189, 237 237, 316 316) and must collapse to one piece.
        let ids = [1024, 1024, 123, 134, 1024, 40, 26, 26, 350, 189, 189, 237, 237,
                   349, 316, 316, 843, 1024, 33, 24, 128, 990, 1024, 62, 6, 385,
                   1024, 768, 1024, 1024, 1002]
        let vocab = sparseVocab([6: "▁the", 24: "▁m", 26: "▁p", 33: "▁and", 40: "▁you",
                                 62: "▁on", 123: "▁C", 128: "il", 134: "ould",
                                 189: "▁up", 237: "▁some", 316: "ff", 349: "▁co",
                                 350: "ick", 385: "▁way", 768: "▁home", 843: "ee",
                                 990: "k", 1002: "?"])
        XCTAssertEqual(CTCDecoder(vocab: vocab).decode(ids),
                       "Could you pick up some coffee and milk on the way home?")
    }

    func testBlankSeparatedRepeatsEmitTwice() {
        // CTC semantics: "aa" is one 'a', but "a␣a" (blank between) is two.
        let decoder = CTCDecoder(vocab: ["▁hi", "!"], blankID: 2)
        XCTAssertEqual(decoder.decode([2, 0, 0, 2, 0, 1, 2]), "hi hi!")
        XCTAssertEqual(decoder.decode([0, 0, 0]), "hi")
        XCTAssertEqual(decoder.decode([]), "")
        XCTAssertEqual(decoder.decode([2, 2, 2]), "", "all blanks decode to nothing")
    }

    func testOutOfRangeIDsAreDropped() {
        let decoder = CTCDecoder(vocab: ["▁ok"], blankID: 1024)
        XCTAssertEqual(decoder.decode([0, 500, -3, 0]), "ok ok",
                       "ids outside the vocab neither crash nor break collapsing")
    }
}
