#!/bin/bash
# Zips the compiled models for a GitHub release: Scripts/package-models.sh 0.2.0 [tiny|base]
set -euo pipefail
VERSION="${1:?version, e.g. 0.2.0}"
MODEL="${2:-tiny}"
case "$MODEL" in tiny|base) ;; *) echo "model must be tiny or base, got: $MODEL" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/build"; [ "$MODEL" != tiny ] && SRC="$ROOT/build-$MODEL"
mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/moonshine-$MODEL-coreml-v$VERSION.zip"
rm -f "$OUT"
( cd "$SRC" && zip -qr "$OUT" Encoder.mlmodelc Decoder.mlmodelc vocab.json )
ls -lh "$OUT"
