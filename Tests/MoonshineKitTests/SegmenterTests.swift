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
