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
        // A short close (3_200 samples of silence) so a two-chunk utterance
        // finalises before its first interim is due — every event below is
        // deterministic, no drain-vs-feed race.
        let live = LiveTranscriber(transcriber: fake, segmenter: Segmenter(silenceToClose: 3_200))
        live.onPartial = { [self] t in lock.lock(); partials.append(t); lock.unlock() }
        live.onFinal = { [self] t in lock.lock(); finals.append(t); lock.unlock() }
        hush(live, chunks: 5)                           // fills the pre-roll
        speak(live, chunks: 2); hush(live, chunks: 2)   // final #1 (4_800 + 1_600 + 3_200)
        waitUntil { self.fake.calls.count == 1 }        // …taken by the drain and blocked
        speak(live, chunks: 2); hush(live, chunks: 2)   // final #2 queued (1_600 + 1_600 + 3_200)
        speak(live, chunks: 2); hush(live, chunks: 2)   // final #3 queued → two waiting
        speak(live, chunks: 8)                          // interim due on the 8th chunk — skipped: 2 finals wait
        fake.release()
        waitUntil { live.isIdle }
        XCTAssertEqual(fake.calls.map(\.count), [9_600, 6_400, 6_400], "three finals, no interim")
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
