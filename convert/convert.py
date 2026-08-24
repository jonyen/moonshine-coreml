"""Converts UsefulSensors/moonshine-tiny to two Core ML models.

Encoder: enumerated audio lengths 1..12 s → encoder_states [1, F, 288] fp16.
Decoder: one token per call, fixed shapes, self-attention KV cache in Core ML
state, argmax inside. See docs/superpowers/specs/2026-08-22-moonshine-watch-captions-design.md.
"""
import argparse, shutil, subprocess
from contextlib import contextmanager
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import MoonshineForConditionalGeneration
from transformers.models.moonshine import modeling_moonshine as _moonshine_hf

# = hf.model.{encoder,decoder}.rotary_emb.inv_freq.numel() for moonshine-tiny, i.e.
# int(hidden_size/num_attention_heads * partial_rotary_factor) // 2 = int(36 * 0.9) // 2 = 16.
# Verified empirically against the loaded model; both encoder and decoder use the same config.
ROTARY_HALF_DIM = 16


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Copy of transformers.models.moonshine.modeling_moonshine.apply_rotary_pos_emb, with its two
    `cos.shape[-1]`-derived slice bounds (the internal `// 2` half-slice and the `rotary_dim` used to
    split q/k into rotated/pass-through parts) replaced by the module-level ROTARY_HALF_DIM constant.
    Both bounds are, in fact, static — a function of config only, independent of audio length — but
    reading them off `cos.shape` at trace time ties them to the encoder's dynamic (EnumeratedShapes)
    sequence-length dimension in coremltools' shape inference, producing an `aten::floor_divide` +
    `aten::Int` pair that fails to convert: 'TypeError: only 0-dimensional arrays can be converted to
    Python scalars'. Passing the bound in as a plain Python int sidesteps this entirely. Used both by
    the encoder (via the monkeypatched MoonshineAttention.forward below) and by the Decoder wrapper's
    own self-attention, which has no dynamic dims but shares this fixed rotary math for consistency."""
    cos = cos.unsqueeze(unsqueeze_dim)[..., :ROTARY_HALF_DIM].repeat_interleave(2, dim=-1)
    sin = sin.unsqueeze(unsqueeze_dim)[..., :ROTARY_HALF_DIM].repeat_interleave(2, dim=-1)
    rotary_dim = ROTARY_HALF_DIM * 2
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([q_rot * cos + _moonshine_hf.rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_embed = torch.cat([k_rot * cos + _moonshine_hf.rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_embed, k_embed


def _traceable_attention_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                                  past_key_values=None, cache_position=None, key_value_states=None, **kwargs):
    """Copy of the installed transformers 4.57.6 `MoonshineAttention.forward`, with the single line
    `bsz, q_len = hidden_states.shape[:-1]` split into two single-dim `.shape[i]` reads. Tracing the
    tuple-slice-then-unpack form emits an aten::size/aten::Int pair that coremltools 9.0 cannot convert
    once the sequence length is a dynamic (EnumeratedShapes/RangeDim) dim: `TypeError: only
    0-dimensional arrays can be converted to Python scalars` inside torch frontend op `_int`. This is
    the only behavioral change from the installed module; monkeypatched onto MoonshineAttention only
    for the duration of `traceable_encoder_attention()` below (affects only calls that go through
    `attn.forward()`, i.e. the encoder — the Decoder wrapper below calls q_proj/k_proj/v_proj/o_proj
    directly and never hits this method)."""
    bsz = hidden_states.shape[0]
    q_len = hidden_states.shape[1]

    query_states = (
        self.q_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    )

    is_cross_attention = key_value_states is not None
    if past_key_values is not None:
        is_updated = past_key_values.is_updated.get(self.layer_idx)
        if is_cross_attention:
            past_key_values.is_updated[self.layer_idx] = True
            past_key_values = past_key_values.cross_attention_cache
        else:
            past_key_values = past_key_values.self_attention_cache

    current_states = key_value_states if key_value_states is not None else hidden_states
    if is_cross_attention and past_key_values and is_updated:
        key_states = past_key_values.layers[self.layer_idx].keys
        value_states = past_key_values.layers[self.layer_idx].values
    else:
        key_states = (
            self.k_proj(current_states).view(bsz, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(current_states).view(bsz, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        if is_cross_attention and past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    if not is_cross_attention:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    attention_interface = _moonshine_hf.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = _moonshine_hf.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    is_causal = self.is_causal and attention_mask is None and q_len > 1

    if self.head_dim_padding > 0:
        query_states = torch.nn.functional.pad(query_states, (0, self.head_dim_padding))
        key_states = torch.nn.functional.pad(key_states, (0, self.head_dim_padding))
        value_states = torch.nn.functional.pad(value_states, (0, self.head_dim_padding))

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, is_causal=is_causal, **kwargs,
    )

    if self.head_dim_padding > 0:
        attn_output = attn_output[..., : -self.head_dim_padding]

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@contextmanager
def traceable_encoder_attention():
    """Monkeypatches MoonshineAttention.forward to _traceable_attention_forward for the duration of
    this context, then restores the original implementation — scoped to the encoder's trace+convert
    only (see convert_encoder below). The patch must not leak process-global: the Decoder path (and
    any stock HF forward/generate call that might share this process) needs to see the original
    installed implementation, not the encoder-only workaround."""
    original = _moonshine_hf.MoonshineAttention.forward
    _moonshine_hf.MoonshineAttention.forward = _traceable_attention_forward
    try:
        yield
    finally:
        _moonshine_hf.MoonshineAttention.forward = original


MODEL_ID = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000
BUCKET_SECONDS = list(range(1, 13))
MAX_FRAMES = 500          # 12 s of audio is 498 encoder frames
MAX_POSITIONS = 194       # bos + 193 tokens (config.max_position_embeddings)
NEG = -1e4                # additive mask value; fp16-safe


class Encoder(torch.nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.enc = hf.model.encoder

    def forward(self, audio):                      # [1, T] float32 in [-1, 1]
        return self.enc(audio).last_hidden_state   # [1, F, 288]


class Decoder(torch.nn.Module):
    """One greedy step. Re-implements the layer math with the HF submodules so
    the trace has no Cache objects and no dynamic slices: the causal mask, the
    encoder padding mask and the one-hot cache write all come from `position`
    and `frames`."""

    def __init__(self, hf):
        super().__init__()
        cfg = hf.config
        self.dec = hf.model.decoder
        self.proj_out = hf.proj_out
        self.heads = cfg.decoder_num_attention_heads
        self.head_dim = cfg.hidden_size // self.heads          # 36; HF pads to 40 for its kernel only
        self.hidden = cfg.hidden_size
        n_layers = len(self.dec.layers)
        shape = (n_layers, 1, self.heads, MAX_POSITIONS, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape))
        self.register_buffer("v_cache", torch.zeros(shape))

    def forward(self, token, encoder_states, frames, position):
        # token [1,1] int32 · encoder_states [1,MAX_FRAMES,288] · frames [1] int32 · position [1] int32
        position = position.long()
        frames = frames.long()
        x = self.dec.embed_tokens(token.long())                                # [1,1,288]
        cos, sin = self.dec.rotary_emb(x, position.view(1, 1))                 # rotary for this one position
        pos = position.view(1, 1, 1, 1)
        tok_idx = torch.arange(MAX_POSITIONS).view(1, 1, 1, MAX_POSITIONS)
        causal = (tok_idx > pos).to(x.dtype) * NEG                             # [1,1,1,P]
        write = (tok_idx == pos).to(x.dtype).view(1, 1, MAX_POSITIONS, 1)      # [1,1,P,1]
        frm_idx = torch.arange(MAX_FRAMES).view(1, 1, 1, MAX_FRAMES)
        enc_mask = (frm_idx >= frames.view(1, 1, 1, 1)).to(x.dtype) * NEG      # [1,1,1,F]

        for i, layer in enumerate(self.dec.layers):
            h = layer.input_layernorm(x)
            x = x + self._self_attn(layer.self_attn, i, h, cos, sin, causal, write)
            h = layer.post_attention_layernorm(x)
            x = x + self._cross_attn(layer.encoder_attn, h, encoder_states, enc_mask)
            h = layer.final_layernorm(x)
            x = x + layer.mlp(h)
        x = self.dec.norm(x)
        logits = self.proj_out(x)                                              # [1,1,32768]
        return torch.argmax(logits, dim=-1).to(torch.int32).view(1)

    def _heads(self, t, n):
        return t.view(1, n, self.heads, self.head_dim).transpose(1, 2)         # [1,H,n,D]

    def _self_attn(self, attn, i, h, cos, sin, causal, write):
        q = self._heads(attn.q_proj(h), 1)
        k = self._heads(attn.k_proj(h), 1)
        v = self._heads(attn.v_proj(h), 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        keep = 1.0 - write
        self.k_cache[i : i + 1] = (self.k_cache[i : i + 1] * keep + k * write)
        self.v_cache[i : i + 1] = (self.v_cache[i : i + 1] * keep + v * write)
        K = self.k_cache[i]                                                    # [1,H,P,D]
        V = self.v_cache[i]
        scores = torch.matmul(q, K.transpose(-1, -2)) * attn.scaling + causal  # [1,H,1,P]
        w = torch.softmax(scores, dim=-1)
        o = torch.matmul(w, V).transpose(1, 2).reshape(1, 1, self.heads * self.head_dim)
        return attn.o_proj(o)

    def _cross_attn(self, attn, h, enc, mask):
        q = self._heads(attn.q_proj(h), 1)
        k = self._heads(attn.k_proj(enc), MAX_FRAMES)
        v = self._heads(attn.v_proj(enc), MAX_FRAMES)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling + mask    # [1,H,1,F]
        w = torch.softmax(scores, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).reshape(1, 1, self.heads * self.head_dim)
        return attn.o_proj(o)


def convert_encoder(hf, out: Path) -> Path:
    enc = Encoder(hf).eval()
    example = torch.zeros(1, 4 * SAMPLE_RATE)
    with traceable_encoder_attention():
        with torch.no_grad():
            traced = torch.jit.trace(enc, example)
        shapes = ct.EnumeratedShapes(shapes=[[1, s * SAMPLE_RATE] for s in BUCKET_SECONDS],
                                     default=[1, 4 * SAMPLE_RATE])
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="audio", shape=shapes, dtype=np.float32)],
            outputs=[ct.TensorType(name="encoder_states", dtype=np.float16)],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )
    model.short_description = "Moonshine Tiny encoder (UsefulSensors/moonshine-tiny), MIT"
    path = out / "Encoder.mlpackage"
    model.save(str(path))
    return path


def convert_decoder(hf, out: Path) -> Path:
    dec = Decoder(hf).eval()
    example = (torch.ones(1, 1, dtype=torch.int32),
               torch.zeros(1, MAX_FRAMES, hf.config.hidden_size),
               torch.tensor([40], dtype=torch.int32),
               torch.tensor([0], dtype=torch.int32))
    with torch.no_grad():
        traced = torch.jit.trace(dec, example)
    cache_shape = tuple(dec.k_cache.shape)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="encoder_states", shape=(1, MAX_FRAMES, hf.config.hidden_size), dtype=np.float16),
            ct.TensorType(name="frames", shape=(1,), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="next_token", dtype=np.int32)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="k_cache"),
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16), name="v_cache"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    model.short_description = "Moonshine Tiny decoder step (UsefulSensors/moonshine-tiny), MIT"
    path = out / "Decoder.mlpackage"
    model.save(str(path))
    return path


def compile_model(package: Path, out: Path) -> Path:
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)], check=True)
    compiled = out / (package.stem + ".mlmodelc")
    assert compiled.is_dir(), compiled
    return compiled


def smoke_test(out: Path) -> None:
    enc = ct.models.MLModel(str(out / "Encoder.mlpackage"))
    dec = ct.models.MLModel(str(out / "Decoder.mlpackage"))
    states = enc.predict({"audio": np.zeros((1, SAMPLE_RATE), dtype=np.float32)})["encoder_states"]
    assert states.shape == (1, 40, 288), states.shape   # 1 s → 40 frames
    padded = np.zeros((1, MAX_FRAMES, 288), dtype=np.float16)
    padded[:, :40] = states
    state = dec.make_state()
    out_token = dec.predict({"token": np.array([[1]], dtype=np.int32), "encoder_states": padded,
                             "frames": np.array([40], dtype=np.int32),
                             "position": np.array([0], dtype=np.int32)}, state=state)["next_token"]
    assert out_token.shape == (1,) and out_token.dtype == np.int32, (out_token.shape, out_token.dtype)
    print(f"smoke test ok: encoder {states.shape}, first decoder token {int(out_token[0])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../build"))
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hf = MoonshineForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    assert hf.model.decoder.rotary_emb.inv_freq.numel() == ROTARY_HALF_DIM, (
        f"ROTARY_HALF_DIM={ROTARY_HALF_DIM} no longer matches this checkpoint's decoder "
        f"rotary_emb.inv_freq (numel={hf.model.decoder.rotary_emb.inv_freq.numel()}); "
        f"update the constant in convert.py"
    )
    if not args.skip_encoder:
        compile_model(convert_encoder(hf, args.out), args.out)
    if not args.skip_decoder:
        compile_model(convert_decoder(hf, args.out), args.out)
    if not (args.out / "vocab.json").exists():
        subprocess.run(["uv", "run", "python", "export_vocab.py", str(args.out / "vocab.json")], check=True)
    smoke_test(args.out)
    print("models in", args.out.resolve())


if __name__ == "__main__":
    main()
