# convert

Converts [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) (Hugging
Face PyTorch) to Core ML: an enumerated-shape encoder and a fixed-shape stateful decoder.

## Setup

```bash
cd convert
uv sync
```

`torch` is pinned to `2.7.0` exactly — the highest version `coremltools 9.0` (also pinned as
`>=8.3`) has been tested against. A newer torch prints a "not tested with coremltools" warning at
import time and is a common source of opaque trace-conversion failures; re-pin and re-sync if that
warning reappears with a future `uv sync`.

## Run

```bash
cd convert
uv run python convert.py --out ../build
```

This downloads (or reuses the local Hugging Face cache for) `UsefulSensors/moonshine-tiny`, traces
the encoder and decoder with `torch.jit.trace`, converts both to Core ML with `coremltools`,
compiles each `.mlpackage` to `.mlmodelc` with `xcrun coremlcompiler`, writes `vocab.json`, and runs
a shape-only smoke test.

`build/` (gitignored) ends up with:

- `Encoder.mlpackage` / `Encoder.mlmodelc`
- `Decoder.mlpackage` / `Decoder.mlmodelc`
- `vocab.json` — a JSON array of 32768 strings, index = token id (also produced standalone by
  `uv run python export_vocab.py ../build/vocab.json`)

Flags: `--skip-encoder` / `--skip-decoder` skip converting that model (useful when iterating on one
side); `vocab.json` is only (re)written if it does not already exist at `--out`.

## Input/output contract

- Encoder: input `audio` float32 `[1, T]`, T ∈ {16000·s | s = 1…12}; output `encoder_states`
  float16 `[1, F, 288]`.
- Decoder: inputs `token` int32 `[1,1]`, `encoder_states` float16 `[1,500,288]`, `frames` int32
  `[1]`, `position` int32 `[1]`; states `k_cache`, `v_cache` float16 `[6,1,8,194,36]`; output
  `next_token` int32 `[1]`.

The decoder runs one greedy step per call: the causal mask, the encoder padding mask, and the
one-hot KV-cache write are all derived inside the model from `frames` and `position`, and argmax
happens inside the model too (`next_token` is already the chosen id, not logits).

## Runtime patches applied in `convert.py`

The installed `transformers` (4.57.6) implementation of `MoonshineAttention.forward` and its
`apply_rotary_pos_emb` helper trip a coremltools 9.0 conversion bug once the encoder's sequence
length is a dynamic (`EnumeratedShapes`) dimension: shape-derived Python ints
(`hidden_states.shape[:-1]`, `cos.shape[-1] // 2`) that are, in fact, static and config-only get
tainted as dynamic by coremltools' shape inference, and the conversion fails with `TypeError: only
0-dimensional arrays can be converted to Python scalars` while converting an `aten::Int` node.
`convert.py` works around this with two functions that are drop-in behavioral copies of the
installed ones, differing only in how they obtain those two static ints — `apply_rotary_pos_emb` at
module scope (used by both the encoder, via a `MoonshineAttention.forward` monkeypatch, and the
Decoder wrapper's own self-attention) and `_traceable_attention_forward` (monkeypatched onto
`MoonshineAttention.forward`, so it only affects the encoder — the Decoder wrapper never calls
`attn.forward()`, only the raw `q_proj`/`k_proj`/`v_proj`/`o_proj`/`scaling` submodules). See the
docstrings on both for the exact diff. `ROTARY_HALF_DIM = 16` was verified empirically against
`hf.model.{encoder,decoder}.rotary_emb.inv_freq.shape` for `moonshine-tiny`; if this ever converts
against a different checkpoint, re-verify that constant first.

`ct.convert()` also prints two harmless messages worth knowing about, not errors:
`Torch var v_cache/k_cache is added again` (informational, from tracing an in-place buffer write
twice — expected for the stateful KV-cache pattern) and an `overflow encountered in cast`
`RuntimeWarning` from a MIL shape-range optimization pass. Separately, loading either compiled model
for prediction (as `smoke_test()` and later the Swift runtime will) prints `E5RT ... MILCompilerForANE
error: failed to compile ANE model using ANEF` to stderr — the Neural Engine compiler rejects some
op in the graph and Core ML falls back to another compute unit automatically; prediction still
succeeds. Actual compute-unit placement and performance are out of scope here — see task A7.

## What the smoke test does and does not prove

`smoke_test()` in `convert.py` only checks output **shapes and dtypes** — that the encoder produces
`(1, 40, 288)` for 1 s of silence and that the decoder's `next_token` is a `(1,)` int32 array. It
does **not** check numerical correctness (e.g. that the decoder actually reproduces the reference
HF model's logits/argmax). That correctness check is `parity_test.py` (task 5), which compares the
compiled Core ML models against the HF PyTorch reference on real audio.
