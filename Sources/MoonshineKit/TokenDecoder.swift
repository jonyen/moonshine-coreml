import Foundation

/// Turns Moonshine token ids back into text. The vocabulary is the 32768
/// strings exported from the Hugging Face `tokenizer.json` (Llama-style BPE:
/// "▁" marks a word boundary, "<0xNN>" is one raw UTF-8 byte).
public struct TokenDecoder {
    public static let bos: Int32 = 1
    public static let eos: Int32 = 2
    /// Ids at or above this are the `<<ST_n>>` stream tokens; 0...2 are
    /// `<unk>`, `<s>`, `</s>`. None of them is text.
    public static let firstSpecialID: Int32 = 32_000

    public let vocab: [String]

    public init(vocab: [String]) {
        self.vocab = vocab
    }

    /// Reads a `vocab.json` array of strings.
    public init(vocabURL: URL) throws {
        vocab = try JSONDecoder().decode([String].self, from: Data(contentsOf: vocabURL))
    }

    public func decode(_ ids: [Int32]) -> String {
        var out = ""
        var bytes: [UInt8] = []
        func flushBytes() {
            guard !bytes.isEmpty else { return }
            out += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll()
        }
        for id in ids {
            guard id > Self.eos, id < Self.firstSpecialID, Int(id) < vocab.count else { continue }
            let token = vocab[Int(id)]
            if let byte = Self.byteFallback(token) {
                bytes.append(byte)
                continue
            }
            flushBytes()
            out += token.replacingOccurrences(of: "▁", with: " ")
        }
        flushBytes()
        return out.hasPrefix(" ") ? String(out.dropFirst()) : out
    }

    /// `"<0x0A>"` → `0x0A`; nil for any other token.
    static func byteFallback(_ token: String) -> UInt8? {
        guard token.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }
}
