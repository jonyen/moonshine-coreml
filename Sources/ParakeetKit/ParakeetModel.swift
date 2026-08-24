import CoreML
import Foundation
import MoonshineKit

/// The Parakeet CTC path: mel + FastConformer encoder (raw audio in) and the
/// CTC head, plus the vocabulary, loaded from one directory — a
/// parakeet-ctc-110m-coreml release unzipped: `Encoder.mlmodelc`,
/// `CTCHead.mlmodelc`, `vocab.json`. Loading takes a second or two; keep the
/// instance for the life of the app.
public final class ParakeetModel {
    public static let sampleRate = 16_000
    /// The encoder's fixed input window: 15 s of 16 kHz audio. Shorter audio
    /// is zero-padded (with the true length passed alongside); longer audio
    /// is truncated. The encode cost is paid for the full window every time.
    public static let windowSamples = 240_000
    /// CTC blank — one past the last vocabulary id.
    public static let blankID = 1024
    /// Encoder output frames for the full window (80 ms per frame).
    public static let maxFrames = 188

    private let encoder: MLModel
    private let ctcHead: MLModel
    public let vocab: [String]

    public init(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        encoder = try MLModel(contentsOf: directory.appendingPathComponent("Encoder.mlmodelc"), configuration: config)
        ctcHead = try MLModel(contentsOf: directory.appendingPathComponent("CTCHead.mlmodelc"), configuration: config)
        vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
    }

    /// Runs mel+encoder and the CTC head on `audio` (Float in -1...1) and
    /// returns the argmax token id per real encoder frame — `blankID` or a
    /// vocabulary index, one per 80 ms — ready for `CTCDecoder.decode`.
    public func frameIDs(_ audio: [Float]) throws -> [Int] {
        let count = min(audio.count, Self.windowSamples)
        let window = try MLMultiArray(shape: [1, NSNumber(value: Self.windowSamples)], dataType: .float32)
        let p = window.dataPointer.bindMemory(to: Float.self, capacity: Self.windowSamples)
        p.initialize(repeating: 0, count: Self.windowSamples)
        if count > 0 {
            audio.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: count) }
        }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: Int32(count))

        let encOut = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: window),
            "audio_length": MLFeatureValue(multiArray: length),
        ]))
        guard let states = encOut.featureValue(for: "encoder")?.multiArrayValue else {
            throw MoonshineError.missingOutput("encoder")
        }
        guard let encoderLength = encOut.featureValue(for: "encoder_length")?.multiArrayValue else {
            throw MoonshineError.missingOutput("encoder_length")
        }

        let ctcOut = try ctcHead.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "encoder": MLFeatureValue(multiArray: states),
        ]))
        guard let logProbs = ctcOut.featureValue(for: "log_probs")?.multiArrayValue else {
            throw MoonshineError.missingOutput("log_probs")
        }
        // log_probs is [1, frames, vocab+blank]; only the first encoder_length
        // frames are real — the rest scored zero padding.
        let frames = min(encoderLength[0].intValue, logProbs.shape[1].intValue)
        return Self.argmaxPerFrame(logProbs, frames: frames)
    }

    /// Argmax over the last axis of a `[1, frames, classes]` array, for the
    /// first `frames` rows.
    static func argmaxPerFrame(_ array: MLMultiArray, frames: Int, classes: Int? = nil) -> [Int] {
        let classes = classes ?? array.shape[2].intValue
        let frameStride = array.strides[1].intValue
        let classStride = array.strides[2].intValue
        var out = [Int](repeating: 0, count: max(frames, 0))
        func run<T: Comparable>(_ base: UnsafePointer<T>) {
            for t in 0..<out.count {
                let row = base + t * frameStride
                var best = 0
                var bestValue = row[0]
                for c in 1..<classes where row[c * classStride] > bestValue {
                    best = c
                    bestValue = row[c * classStride]
                }
                out[t] = best
            }
        }
        switch array.dataType {
        case .float32:
            run(array.dataPointer.bindMemory(to: Float.self, capacity: frames * frameStride))
        case .float16:
            run(array.dataPointer.bindMemory(to: Float16.self, capacity: frames * frameStride))
        default:
            for t in 0..<out.count {   // slow path: Core ML handed back another dtype
                var best = 0
                var bestValue = array[[0, NSNumber(value: t), 0]].doubleValue
                for c in 1..<classes {
                    let v = array[[0, NSNumber(value: t), NSNumber(value: c)]].doubleValue
                    if v > bestValue { best = c; bestValue = v }
                }
                out[t] = best
            }
        }
        return out
    }
}
