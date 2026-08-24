import CoreML
import Foundation

/// One utterance in, text out: Int16 → Float, encode, greedy decode until eos
/// or the token cap, then `TokenDecoder`. One request at a time — callers
/// serialise (LiveTranscriber does).
public final class Transcriber: Transcribing {
    private let model: MoonshineModel
    private let decoder: TokenDecoder

    public init(model: MoonshineModel) {
        self.model = model
        self.decoder = TokenDecoder(vocab: model.vocab)
    }

    /// Moonshine's rule of thumb: at most 6.5 tokens per second of audio, plus
    /// a little slack, never more than the 193 positions left after bos.
    public static func maxTokens(forSamples count: Int) -> Int {
        let seconds = Double(count) / Double(MoonshineModel.sampleRate)
        return min(MoonshineModel.maxPositions - 1, Int(seconds * 6.5) + 2)
    }

    public func transcribe(_ samples: [Int16]) throws -> String {
        guard !samples.isEmpty else { return "" }
        let audio = samples.map { Float($0) / 32_768 }
        let encoded = try model.encode(audio)
        let state = model.makeDecoderState()
        var ids: [Int32] = []
        var token = TokenDecoder.bos
        for position in 0..<Self.maxTokens(forSamples: samples.count) {
            let next = try model.decodeStep(token: token, encoded: encoded, position: position, state: state)
            if next == TokenDecoder.eos { break }
            ids.append(next)
            token = next
        }
        return decoder.decode(ids)
    }
}
