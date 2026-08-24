import Foundation
import MoonshineKit

/// One utterance in, text out: Int16 → Float, fixed-window encode, CTC head,
/// greedy decode. Conforms to MoonshineKit's `Transcribing`, so it drops into
/// the existing `Segmenter`/`LiveTranscriber` pipeline unchanged. One request
/// at a time — callers serialise (LiveTranscriber does).
///
/// Unlike Moonshine, Parakeet emits punctuation and capitalisation.
public final class ParakeetTranscriber: Transcribing {
    private let model: ParakeetModel
    private let decoder: CTCDecoder

    public init(model: ParakeetModel) {
        self.model = model
        self.decoder = CTCDecoder(vocab: model.vocab)
    }

    public func transcribe(_ samples: [Int16]) throws -> String {
        guard !samples.isEmpty else { return "" }
        let audio = samples.map { Float($0) / 32_768 }
        return decoder.decode(try model.frameIDs(audio))
    }
}
