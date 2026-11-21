"""Lesson 8: export a checkpoint as a GGUF file and an Ollama model.

  uv run --extra export python -m theog.export --ckpt runs/theog-50m/best.pt \
      --tokenizer data/tokenizer.json --out release/theog-50m-Q8_0.gguf --quant Q8_0
  ollama create theog:no_think -f release/Modelfile.no_think

The model is Qwen3's layout (lesson 4), so the GGUF says `qwen3` and every
runtime that runs Qwen3 (llama.cpp, Ollama, LM Studio) runs it unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARCH = "qwen3"
# course name under blocks.{i}. -> GGUF name under blk.{i}.
PER_BLOCK = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.q.weight": "attn_q.weight",
    "attn.k.weight": "attn_k.weight",
    "attn.v.weight": "attn_v.weight",
    "attn.o.weight": "attn_output.weight",
    "attn.q_norm.weight": "attn_q_norm.weight",
    "attn.k_norm.weight": "attn_k_norm.weight",
    "mlp_norm.weight": "ffn_norm.weight",
    "mlp.gate.weight": "ffn_gate.weight",
    "mlp.up.weight": "ffn_up.weight",
    "mlp.down.weight": "ffn_down.weight",
}

# The chat format of lesson 3 as a Jinja template (llama.cpp, LM Studio).
# A user message ending in "/think" opens the think span; anything else is /no_think.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "{%- set text = message['content'] | trim -%}"
    "{%- set think = text.endswith('/think') and not text.endswith('/no_think') -%}"
    "{%- if text.endswith('/no_think') -%}{%- set text = text[:-9] | trim -%}"
    "{%- elif think -%}{%- set text = text[:-6] | trim -%}{%- endif -%}"
    "{{- '<|User|>\\n' + text + (' /think' if think else ' /no_think') + '\\n<|/User|><|Assistant|>' -}}"
    "{%- if loop.last and add_generation_prompt -%}"
    "{{- '<|Think|>' if think else '<|Think|><|/Think|><|Answer|>' -}}"
    "{%- endif -%}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{- '<|Think|><|/Think|><|Answer|>' + (message['content'] | trim) + '<|/Answer|><|end|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
)

SAMPLING = {"temperature": 0.2, "top_k": 40, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
            "repeat_last_n": 64}
STOPS = ["<|/Answer|>", "<|end|>", "<|User|>"]


def tensors(model) -> list[tuple[str, np.ndarray]]:
    """Every weight as (GGUF name, float32 numpy array), in this order:
    token_embd.weight, then for each block i every PER_BLOCK entry as
    blk.{i}.{gguf name}, then output_norm.weight.
    """
    # EXERCISE 8.1: lessons/08-export/README.md
    # Hint: model.state_dict() keys are the course names: 'embed.weight',
    #       'blocks.0.attn.q.weight', ..., 'norm.weight'. .detach().float().cpu().numpy().
    raise NotImplementedError("exercise 8.1: write tensors")


def add_tokenizer(w, tokenizer: Path, vocab_size: int) -> None:
    import gguf
    from theog.tokens import ANSWER_CLOSE, END, PAD, USER
    tok = json.loads(tokenizer.read_text())
    vocab = {i: t for t, i in tok["model"]["vocab"].items()}
    special = {a["id"] for a in tok["added_tokens"]}
    names, types = [], []
    for i in range(vocab_size):
        if i in vocab:
            names.append(vocab[i])
            types.append(gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL)
        else:  # the model has more rows than the tokenizer has tokens
            names.append(f"<|unused_{i}|>")
            types.append(gguf.TokenType.UNUSED)
    w.add_tokenizer_model("gpt2")   # byte-level BPE
    w.add_tokenizer_pre("gpt-2")    # with GPT-2's split regex, as in lesson 1
    w.add_token_list(names)
    w.add_token_types(types)
    w.add_token_merges([" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]])
    w.add_bos_token_id(USER)
    w.add_add_bos_token(False)
    w.add_eos_token_id(ANSWER_CLOSE)
    w.add_eot_token_id(END)
    w.add_pad_token_id(PAD)
    w.add_chat_template(CHAT_TEMPLATE)


def export_gguf(model, tokenizer: Path, out: Path, quant: str = "F16", name: str = "theOG") -> str:
    import gguf
    cfg = model.cfg
    spec = getattr(model, "attn_spec", {})
    if set(spec) - {"kv"}:
        raise SystemExit(f"attention {spec} is not in llama.cpp's qwen3 layout; only kv= exports")
    qtype = {"F16": gguf.GGMLQuantizationType.F16, "Q8_0": gguf.GGMLQuantizationType.Q8_0}[quant]
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), ARCH)
    w.add_name(name)
    w.add_file_type({"F16": gguf.LlamaFileType.MOSTLY_F16, "Q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[quant])
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.hidden)
    w.add_block_count(cfg.layers)
    w.add_feed_forward_length(cfg.ffn)
    w.add_head_count(cfg.heads)
    w.add_head_count_kv(spec.get("kv", cfg.heads))
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.eps)
    add_tokenizer(w, tokenizer, cfg.vocab_size)
    for n, a in tensors(model):
        if a.ndim == 1:      # norm weights stay float32: tiny, and precision matters
            w.add_tensor(n, a.astype(np.float32))
        elif quant == "F16":
            w.add_tensor(n, a.astype(np.float16))
        else:
            w.add_tensor(n, gguf.quants.quantize(a, qtype), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return hashlib.sha256(out.read_bytes()).hexdigest()


def go_template(mode: str) -> str:
    """Ollama's template language has no string functions, so the mode is
    fixed per Modelfile."""
    opener = "<|Think|>" if mode == "think" else "<|Think|><|/Think|><|Answer|>"
    return ('{{- range $i, $m := .Messages }}'
            '{{- if eq $m.Role "user" }}<|User|>\n{{ $m.Content }} /' + mode + '\n<|/User|><|Assistant|>'
            '{{- if eq (len (slice $.Messages $i)) 1 }}' + opener + '{{ end }}'
            '{{- else if eq $m.Role "assistant" }}<|Think|><|/Think|><|Answer|>{{ $m.Content }}<|/Answer|><|end|>'
            '{{- end }}{{- end }}')


def modelfile(gguf_path: str, mode: str = "no_think", num_ctx: int = 2048) -> str:
    """The text of an Ollama Modelfile: FROM the gguf path, the TEMPLATE
    from go_template(mode) in triple quotes, one PARAMETER stop line per STOPS
    entry, one PARAMETER line per SAMPLING entry, then PARAMETER num_ctx
    {num_ctx}: the model's context, so Ollama never feeds it a longer window
    than it trained on.
    """
    # EXERCISE 8.2: lessons/08-export/README.md
    # Hint: Lines like: FROM ./m.gguf / TEMPLATE """...""" / PARAMETER stop "<|end|>" /
    #       PARAMETER temperature 0.2. Join with newlines, end with one.
    raise NotImplementedError("exercise 8.2: write modelfile")


def main() -> None:
    from theog.train import load_model
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quant", choices=["F16", "Q8_0"], default="Q8_0")
    ap.add_argument("--name", default="theOG")
    a = ap.parse_args()
    model = load_model(a.ckpt)
    sha = export_gguf(model, a.tokenizer, a.out, a.quant, a.name)
    print(f"{a.out}  {a.quant}  sha256 {sha}")
    for mode in ("no_think", "think"):
        mf = a.out.parent / f"Modelfile.{mode}"
        mf.write_text(modelfile(f"./{a.out.name}", mode, model.cfg.max_seq_len))
        print(f"{mf}  ->  ollama create theog:{mode} -f {mf}")


if __name__ == "__main__":
    main()
