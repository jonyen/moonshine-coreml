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
///
/// `onPartial`, `onFinal` and `onError` are invoked on the internal inference
/// queue — hop to the main actor before touching UI.
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

    /// One pool per job. Core ML returns its prediction outputs autoreleased,
    /// and a GCD block only drains its pool when the block returns. In
    /// continuous speech this loop never returns, so without a pool here
    /// every job's outputs (about 1.25 MB for Moonshine Base) stay alive
    /// until the speaker pauses. On a watch that reached the 300 MB jetsam
    /// limit within minutes.
    private func drain() {
        while let job = nextJob() {
            autoreleasepool { run(job) }
        }
    }

    private func run(_ job: Job) {
        if !job.isFinal, !segmentIsOpen(job.segment, generation: job.generation) { return }
        do {
            let text = try transcriber.transcribe(job.audio)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, generationMatches(job.generation) else { return }
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
