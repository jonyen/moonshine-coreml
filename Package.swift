// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MoonshineKit",
    platforms: [.watchOS(.v11), .iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "MoonshineKit", targets: ["MoonshineKit"]),
        .executable(name: "moonshine-bench", targets: ["moonshine-bench"]),
    ],
    targets: [
        .target(name: "MoonshineKit"),
        .executableTarget(name: "moonshine-bench", dependencies: ["MoonshineKit"]),
        .testTarget(name: "MoonshineKitTests", dependencies: ["MoonshineKit"]),
    ],
    swiftLanguageModes: [.v5]
)
