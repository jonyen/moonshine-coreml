#!/usr/bin/env python3
"""Produce int8 weight-quantized variants of the Parakeet CTC-path mlpackages.

The OpenVoiceOS packages already store FLOAT16 weights (despite metadata.json
saying FLOAT32 — see README), so a FLOAT16 re-conversion is a no-op. The size
reduction that actually exists is int8: per-channel linear-symmetric weight
quantization, weights dequantized to fp16 at run time. Outputs land in
models/int8/ with the same file names and I/O contract.

Usage:
    .venv-parakeet/bin/python make_int8.py

Verify parity afterwards with:
    .venv-parakeet/bin/python run_ctc.py --precision int8
"""
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

HERE = Path(__file__).parent
MODELS = HERE / "models"
OUT = MODELS / "int8"

PACKAGES = ["parakeet_mel_encoder.mlpackage", "parakeet_ctc_decoder.mlpackage"]


def dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    OUT.mkdir(exist_ok=True)
    config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    for name in PACKAGES:
        src, dst = MODELS / name, OUT / name
        model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)
        quantized = cto.linear_quantize_weights(model, config=config)
        if dst.exists():
            shutil.rmtree(dst)
        quantized.save(str(dst))
        print(f"{name}: {dir_mb(src):.1f} MB fp16 -> {dir_mb(dst):.1f} MB int8")
    print(f"int8 packages in {OUT}")


if __name__ == "__main__":
    main()
