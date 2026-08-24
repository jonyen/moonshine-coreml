"""Writes vocab.json: a JSON array of 32768 strings, index = token id."""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "UsefulSensors/moonshine-tiny"
VOCAB_SIZE = 32768

def main(out: Path) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab = [""] * VOCAB_SIZE
    for token, idx in tok.get_vocab().items():
        if idx < VOCAB_SIZE:
            vocab[idx] = token
    assert vocab[1] == "<s>" and vocab[2] == "</s>", (vocab[:3])
    assert vocab[32000].startswith("<<ST_"), vocab[32000]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False))
    filled = sum(1 for v in vocab if v)
    print(f"wrote {out} ({filled} of {VOCAB_SIZE} ids named)")

if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "../build/vocab.json"))
