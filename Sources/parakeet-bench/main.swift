import Foundation
import MoonshineKit
import ParakeetKit

// parakeet-bench --models DIR file.wav [--cpu]
//   Transcribe the file once cold, then five warm runs; print text + timings.
//   Every run pays for the full fixed 15 s window, so warm ms is what an
//   interim transcription costs regardless of clip length.
//   --cpu: computeUnits = .cpuOnly (what the watch runs today).

var args = Array(CommandLine.arguments.dropFirst())
func take(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...i + 1); return v
}
func has(_ flag: String) -> Bool {
    guard let i = args.firstIndex(of: flag) else { return false }
    args.remove(at: i); return true
}
guard let modelsDir = take("--models") else {
    FileHandle.standardError.write(Data("usage: parakeet-bench --models DIR file.wav [--cpu]\n".utf8)); exit(2)
}
let cpu = has("--cpu")
guard let file = args.first else { FileHandle.standardError.write(Data("missing wav\n".utf8)); exit(2) }

func ms(_ block: () throws -> Void) rethrows -> Double {
    let t = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
}

do {
    let samples = try WAVReader.readInt16Mono16k(URL(fileURLWithPath: file))
    let seconds = Double(samples.count) / 16_000
    var model: ParakeetModel!
    let load = try ms { model = try ParakeetModel(directory: URL(fileURLWithPath: modelsDir), computeUnits: cpu ? .cpuOnly : .all) }
    print(String(format: "loaded in %.0f ms (%@), clip %.2f s", load, cpu ? "cpu" : "all", seconds))
    let transcriber = ParakeetTranscriber(model: model)

    var text = ""
    let cold = try ms { text = try transcriber.transcribe(samples) }
    var warms: [Double] = []
    for _ in 0..<5 { warms.append(try ms { text = try transcriber.transcribe(samples) }) }
    let warm = warms.reduce(0, +) / Double(warms.count)
    print("text: \(text)")
    print(String(format: "cold %.0f ms, warm %.1f ms (min %.1f, max %.1f over 5), RTF %.3f (warm)",
                 cold, warm, warms.min()!, warms.max()!, warm / 1000 / seconds))
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8)); exit(1)
}
