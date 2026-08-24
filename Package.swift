// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MoonshineKit",
    platforms: [.watchOS(.v11), .iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "MoonshineKit", targets: ["MoonshineKit"]),
        .library(name: "ParakeetKit", targets: ["ParakeetKit"]),
        .executable(name: "moonshine-bench", targets: ["moonshine-bench"]),
        .executable(name: "parakeet-bench", targets: ["parakeet-bench"]),
    ],
    targets: [
        .target(name: "MoonshineKit"),
        .target(name: "ParakeetKit", dependencies: ["MoonshineKit"]),
        .executableTarget(name: "moonshine-bench", dependencies: ["MoonshineKit"]),
        .executableTarget(name: "parakeet-bench", dependencies: ["MoonshineKit", "ParakeetKit"]),
        .testTarget(name: "MoonshineKitTests", dependencies: ["MoonshineKit"]),
        .testTarget(name: "ParakeetKitTests", dependencies: ["ParakeetKit", "MoonshineKit"]),
    ],
    swiftLanguageModes: [.v5]
)
