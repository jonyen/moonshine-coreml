#!/bin/bash
# Zips the compiled models for a GitHub release: Scripts/package-models.sh 0.1.0
set -euo pipefail
VERSION="${1:?version, e.g. 0.1.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/moonshine-tiny-coreml-v$VERSION.zip"
rm -f "$OUT"
( cd "$ROOT/build" && zip -qr "$OUT" Encoder.mlmodelc Decoder.mlmodelc vocab.json )
ls -lh "$OUT"
