#!/bin/bash
# Downloads a release's compiled models: Scripts/fetch-models.sh VERSION [tiny|base|parakeet] [DEST]
# The model defaults to tiny; DEST defaults to build/ for tiny, build-<model>/ otherwise.
set -euo pipefail
VERSION="${1:?version, e.g. 0.2.0}"
shift
MODEL="tiny"
case "${1:-}" in tiny|base|parakeet) MODEL="$1"; shift ;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_DEST="$ROOT/build"; [ "$MODEL" != tiny ] && DEFAULT_DEST="$ROOT/build-$MODEL"
DEST="${1:-$DEFAULT_DEST}"
ASSET="moonshine-$MODEL-coreml-v$VERSION.zip"
[ "$MODEL" = parakeet ] && ASSET="parakeet-ctc-110m-coreml-v$VERSION.zip"
URL="https://github.com/jonyen/moonshine-coreml/releases/download/v$VERSION/$ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL "$URL" -o "$TMP/models.zip"
mkdir -p "$DEST"
unzip -oq "$TMP/models.zip" -d "$DEST"
echo "models in $DEST:"; ls "$DEST"
