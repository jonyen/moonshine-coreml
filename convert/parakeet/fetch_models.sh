#!/usr/bin/env bash
# Fetch the Core ML conversion of nvidia/parakeet-tdt_ctc-110m (CTC path only)
# from OpenVoiceOS/parakeet-tdt-ctc-110m-coreml on Hugging Face (CC-BY-4.0),
# and set up the spike venv. ~220 MB download.
set -euo pipefail
cd "$(dirname "$0")"

uv venv .venv-parakeet --python 3.11
uv pip install --python .venv-parakeet/bin/python coremltools numpy huggingface_hub

.venv-parakeet/bin/hf download OpenVoiceOS/parakeet-tdt-ctc-110m-coreml \
  --include "parakeet_mel_encoder.mlpackage/*" "parakeet_ctc_decoder.mlpackage/*" \
            "vocab.json" "metadata.json" "infer.py" "README.md" \
  --local-dir models

echo "Done. Run: .venv-parakeet/bin/python run_ctc.py"
