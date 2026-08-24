#!/bin/bash
# Synthesises the short clips the tests and the parity check run on.
# macOS only: uses the system voice. Re-running overwrites the wavs.
set -euo pipefail
cd "$(dirname "$0")/../test-assets"
make() {
  local name="$1" text="$2"
  say -o "tmp.aiff" "$text"
  afconvert -f WAVE -d LEI16@16000 -c 1 tmp.aiff "$name.wav"
  rm tmp.aiff
  printf '%s\n' "$text" > "$name.say.txt"
}
make hello   "Hello, this is a test of captions running on the watch."
make weather "The weather tomorrow looks cold, with a chance of rain in the afternoon."
make coffee  "Could you pick up some coffee and milk on the way home?"
echo "wrote: $(ls *.wav | tr '\n' ' ')"
