#!/bin/bash
# Zips the compiled models for a GitHub release: Scripts/package-models.sh 0.3.0 [tiny|base|parakeet]
#
# tiny/base zip an existing build/ (or build-base/) as moonshine-<model>-coreml-v<version>.zip.
# parakeet compiles the mlpackages in convert/parakeet/models/ (fetched by
# convert/parakeet/fetch_models.sh; int8/ produced by make_int8.py) into
# build-parakeet/ and zips it as parakeet-ctc-110m-coreml-v<version>.zip with
# Encoder.mlmodelc, CTCHead.mlmodelc and vocab.json at the zip root (the fp16
# build) plus a self-contained int8/ variant.
set -euo pipefail
VERSION="${1:?version, e.g. 0.3.0}"
MODEL="${2:-tiny}"
case "$MODEL" in tiny|base|parakeet) ;; *) echo "model must be tiny, base or parakeet, got: $MODEL" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/dist"

if [ "$MODEL" = parakeet ]; then
    PKGS="$ROOT/convert/parakeet/models"
    BUILD="$ROOT/build-parakeet"
    compile_pair() {  # compile_pair SRC_DIR DEST_DIR
        mkdir -p "$2"
        rm -rf "$2/Encoder.mlmodelc" "$2/CTCHead.mlmodelc"
        xcrun coremlcompiler compile "$1/parakeet_mel_encoder.mlpackage" "$2" > /dev/null
        xcrun coremlcompiler compile "$1/parakeet_ctc_decoder.mlpackage" "$2" > /dev/null
        mv "$2/parakeet_mel_encoder.mlmodelc" "$2/Encoder.mlmodelc"
        mv "$2/parakeet_ctc_decoder.mlmodelc" "$2/CTCHead.mlmodelc"
        cp "$PKGS/vocab.json" "$2/vocab.json"
    }
    compile_pair "$PKGS" "$BUILD"
    compile_pair "$PKGS/int8" "$BUILD/int8"
    OUT="$ROOT/dist/parakeet-ctc-110m-coreml-v$VERSION.zip"
    rm -f "$OUT"
    ( cd "$BUILD" && zip -qr "$OUT" Encoder.mlmodelc CTCHead.mlmodelc vocab.json int8 )
else
    SRC="$ROOT/build"; [ "$MODEL" != tiny ] && SRC="$ROOT/build-$MODEL"
    OUT="$ROOT/dist/moonshine-$MODEL-coreml-v$VERSION.zip"
    rm -f "$OUT"
    ( cd "$SRC" && zip -qr "$OUT" Encoder.mlmodelc Decoder.mlmodelc vocab.json )
fi
ls -lh "$OUT"
