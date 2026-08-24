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
