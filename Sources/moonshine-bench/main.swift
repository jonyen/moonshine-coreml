import Foundation
import MoonshineKit

// moonshine-bench --models DIR file.wav [--live] [--cpu]
//   Default: transcribe the file twice (cold, warm) and print text + timings.
//   --live: push the file through LiveTranscriber in 100 ms chunks and print partials/finals.
//   --cpu:  computeUnits = .cpuOnly, to compare against the Neural Engine.

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
    FileHandle.standardError.write(Data("usage: moonshine-bench --models DIR file.wav [--live] [--cpu]\n".utf8)); exit(2)
}
let live = has("--live"), cpu = has("--cpu")
guard let file = args.first else { FileHandle.standardError.write(Data("missing wav\n".utf8)); exit(2) }

func ms(_ block: () throws -> Void) rethrows -> Double {
    let t = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
}

do {
    let samples = try WAVReader.readInt16Mono16k(URL(fileURLWithPath: file))
    let seconds = Double(samples.count) / 16_000
    var model: MoonshineModel!
    let load = try ms { model = try MoonshineModel(directory: URL(fileURLWithPath: modelsDir), computeUnits: cpu ? .cpuOnly : .all) }
    print(String(format: "loaded in %.0f ms (%@), clip %.2f s", load, cpu ? "cpu" : "all", seconds))
    let transcriber = Transcriber(model: model)

    if live {
        let liveT = LiveTranscriber(transcriber: transcriber)
        let start = DispatchTime.now().uptimeNanoseconds
        func stamp() -> String { String(format: "%7.0f ms", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
        liveT.onPartial = { print("\(stamp())  partial: \($0)") }
        liveT.onFinal = { print("\(stamp())  FINAL:   \($0)") }
        liveT.onError = { print("\(stamp())  error:   \($0)") }
        var i = 0
        while i < samples.count {
            liveT.feed(Array(samples[i..<min(i + 1_600, samples.count)]))
            i += 1_600
            Thread.sleep(forTimeInterval: 0.1)   // real time, so the timings mean something
        }
        liveT.flush()
        while !liveT.isIdle { Thread.sleep(forTimeInterval: 0.01) }
        print("\(stamp())  done")
    } else {
        var text = ""
        let cold = try ms { text = try transcriber.transcribe(samples) }
        let warm = try ms { text = try transcriber.transcribe(samples) }
        print("text: \(text)")
        print(String(format: "cold %.0f ms, warm %.0f ms, RTF %.2f (warm)", cold, warm, warm / 1000 / seconds))
        // Break the warm run down.
        let audio = samples.map { Float($0) / 32_768 }
        var encoded: MoonshineModel.Encoded!
        let enc = try ms { encoded = try model.encode(audio) }
        let state = model.makeDecoderState()
        var token: Int32 = 1, steps = 0
        let dec = try ms {
            for p in 0..<Transcriber.maxTokens(forSamples: samples.count) {
                token = try model.decodeStep(token: token, encoded: encoded, position: p, state: state)
                steps += 1
                if token == 2 { break }
            }
        }
        print(String(format: "encode %.0f ms, decode %.0f ms for %d steps (%.1f ms/token)", enc, dec, steps, dec / Double(max(steps, 1))))
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8)); exit(1)
}
