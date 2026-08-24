import Foundation

public enum MoonshineError: Error, CustomStringConvertible {
    case missingOutput(String)
    case unsupportedWAV(String)
    public var description: String {
        switch self {
        case .missingOutput(let name): return "Core ML output '\(name)' missing"
        case .unsupportedWAV(let why): return "unsupported WAV: \(why)"
        }
    }
}

/// Minimal RIFF reader for the one format MoonshineKit speaks: 16-bit PCM, mono, 16 kHz.
public enum WAVReader {
    public static func readInt16Mono16k(_ url: URL) throws -> [Int16] {
        let data = try Data(contentsOf: url)
        guard data.count > 12, data[0..<4] == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) else {
            throw MoonshineError.unsupportedWAV("not a RIFF/WAVE file")
        }
        var offset = 12
        var format: (channels: Int, rate: Int, bits: Int, pcm: Bool)?
        while offset + 8 <= data.count {
            let id = String(decoding: data[offset..<offset + 4], as: UTF8.self)
            let size = Int(data.readUInt32LE(at: offset + 4))
            let body = offset + 8
            if id == "fmt " {
                guard body + 16 <= data.count else { throw MoonshineError.unsupportedWAV("short fmt chunk") }
                format = (channels: Int(data.readUInt16LE(at: body + 2)),
                          rate: Int(data.readUInt32LE(at: body + 4)),
                          bits: Int(data.readUInt16LE(at: body + 14)),
                          pcm: data.readUInt16LE(at: body) == 1)
            } else if id == "data" {
                guard let f = format else { throw MoonshineError.unsupportedWAV("data before fmt") }
                guard f.pcm, f.bits == 16, f.channels == 1, f.rate == 16_000 else {
                    throw MoonshineError.unsupportedWAV("need 16-bit PCM mono 16 kHz, got \(f)")
                }
                let end = min(body + size, data.count)
                let count = (end - body) / 2
                var samples = [Int16](repeating: 0, count: count)
                _ = samples.withUnsafeMutableBytes { data.copyBytes(to: $0, from: body..<(body + count * 2)) }
                return samples
            }
            offset = body + size + (size & 1)
        }
        throw MoonshineError.unsupportedWAV("no data chunk")
    }
}

private extension Data {
    func readUInt16LE(at i: Int) -> UInt16 { UInt16(self[i]) | UInt16(self[i + 1]) << 8 }
    func readUInt32LE(at i: Int) -> UInt32 {
        UInt32(self[i]) | UInt32(self[i + 1]) << 8 | UInt32(self[i + 2]) << 16 | UInt32(self[i + 3]) << 24
    }
}
