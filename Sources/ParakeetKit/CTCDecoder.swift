import Foundation

/// Greedy CTC decode over Parakeet's 1024-entry SentencePiece vocabulary:
/// collapse repeated ids, drop blanks, look the survivors up in the piece
/// list, `▁` → space. The vocabulary has no `<0xNN>` byte-fallback pieces, so
/// id → string lookup is the whole detokenizer (this mirrors the reference
/// decode in `convert/parakeet/run_ctc.py` exactly).
public struct CTCDecoder {
    public let vocab: [String]
    public let blankID: Int

    public init(vocab: [String], blankID: Int = ParakeetModel.blankID) {
        self.vocab = vocab
        self.blankID = blankID
    }

    /// `frameIDs` is one argmax id per encoder frame (`ParakeetModel.frameIDs`).
    public func decode(_ frameIDs: [Int]) -> String {
        var out = ""
        var prev = -1
        for id in frameIDs {
            if id != blankID, id != prev, id >= 0, id < vocab.count {
                out += vocab[id]
            }
            prev = id
        }
        return out.replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
