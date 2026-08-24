import CoreML
import Foundation

/// The two compiled Core ML models plus the vocabulary, loaded from one
/// directory — a moonshine-coreml release unzipped: `Encoder.mlmodelc`,
/// `Decoder.mlmodelc`, `vocab.json`. Loading takes a second or two; keep the
/// instance for the life of the app.
public final class MoonshineModel {
    public static let sampleRate = 16_000
    /// Encoder input buckets, in seconds. Audio is zero-padded up to the next
    /// bucket; anything longer than the last is truncated to it.
    public static let bucketSeconds = Array(1...12)
    /// The decoder's fixed encoder-state length; 12 s of audio is 498 frames.
    public static let maxFrames = 500
    public static let hiddenSize = 288
    /// Decoder positions: bos plus up to 193 generated tokens.
    public static let maxPositions = 194

    public struct Encoded {
        /// `[1, maxFrames, hiddenSize]` float16, zero beyond `frames`.
        public let states: MLMultiArray
        public let frames: Int
    }

    private let encoder: MLModel
    private let decoder: MLModel
    public let vocab: [String]

    public init(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        encoder = try MLModel(contentsOf: directory.appendingPathComponent("Encoder.mlmodelc"), configuration: config)
        decoder = try MLModel(contentsOf: directory.appendingPathComponent("Decoder.mlmodelc"), configuration: config)
        vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
    }

    /// Runs the encoder on `audio` (Float in -1...1) padded to its bucket, and
    /// lays the result into the decoder's fixed-size input. Empty audio
    /// encodes as one second of silence (the padded buffer is already zeroed).
    public func encode(_ audio: [Float]) throws -> Encoded {
        let seconds = min(Self.bucketSeconds.last!,
                          max(1, Int((Double(audio.count) / Double(Self.sampleRate)).rounded(.up))))
        let length = seconds * Self.sampleRate
        let input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float32)
        let p = input.dataPointer.bindMemory(to: Float.self, capacity: length)
        p.initialize(repeating: 0, count: length)
        let n = min(audio.count, length)
        if n > 0 {
            audio.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: n) }
        }

        let out = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": MLFeatureValue(multiArray: input)]))
        guard let states = out.featureValue(for: "encoder_states")?.multiArrayValue else {
            throw MoonshineError.missingOutput("encoder_states")
        }
        let frames = min(states.shape[1].intValue, Self.maxFrames)
        let padded = try MLMultiArray(shape: [1, NSNumber(value: Self.maxFrames), NSNumber(value: Self.hiddenSize)], dataType: .float16)
        let total = Self.maxFrames * Self.hiddenSize
        memset(padded.dataPointer, 0, total * MemoryLayout<Float16>.size)
        let used = frames * Self.hiddenSize
        if states.dataType == .float16, Self.isContiguous(states) {
            memcpy(padded.dataPointer, states.dataPointer, used * MemoryLayout<Float16>.size)
        } else {
            for i in 0..<used { padded[i] = states[i] }   // slow path: Core ML handed back another layout
        }
        return Encoded(states: padded, frames: frames)
    }

    public func makeDecoderState() -> MLState { decoder.makeState() }

    /// One greedy step. `position` is the index of `token` in the sequence (0 for bos).
    public func decodeStep(token: Int32, encoded: Encoded, position: Int, state: MLState) throws -> Int32 {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: token)
        let frames = try MLMultiArray(shape: [1], dataType: .int32)
        frames[0] = NSNumber(value: Int32(encoded.frames))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenArray),
            "encoder_states": MLFeatureValue(multiArray: encoded.states),
            "frames": MLFeatureValue(multiArray: frames),
            "position": MLFeatureValue(multiArray: pos),
        ])
        let out = try decoder.prediction(from: input, using: state)
        guard let next = out.featureValue(for: "next_token")?.multiArrayValue else {
            throw MoonshineError.missingOutput("next_token")
        }
        return next[0].int32Value
    }

    private static func isContiguous(_ array: MLMultiArray) -> Bool {
        var expected = 1
        for (shape, stride) in zip(array.shape.reversed(), array.strides.reversed()) {
            if stride.intValue != expected { return false }
            expected *= shape.intValue
        }
        return true
    }
}
