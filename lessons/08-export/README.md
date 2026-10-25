# Lesson 8: export to GGUF and Ollama

A `.pt` checkpoint needs PyTorch and this repo. A **GGUF** file holds everything a runtime needs in
one file: the weights, the tokenizer, the chat template and the model's shape. llama.cpp, Ollama and
LM Studio load it directly, on a laptop, with no Python.

## What goes into the file: `theog/export.py`

**1. The architecture name and shape.** The file says `qwen3` (lesson 4 explains why) and records
the numbers the runtime needs to build the model: 10 blocks, hidden 512, 4 heads of 128, ffn 2048,
context 2048, RoPE base 10000, norm eps 1e-5.

**2. The weights, renamed.** `PER_BLOCK` maps each course name to the name llama.cpp expects:

```
blocks.3.attn.q.weight   ->  blk.3.attn_q.weight
blocks.3.mlp.down.weight ->  blk.3.ffn_down.weight
embed.weight             ->  token_embd.weight    (also the output head: tied)
```

`tensors()` asserts that every weight found a slot. A missing weight would otherwise be silently
left out of the file.

**3. Quantization.** `F16` stores each weight in 16 bits instead of 32. `Q8_0` stores blocks of 32
weights as 8-bit integers plus one scale per block, which is about half the size of F16. theOG ships
Q8_0. The norm weights stay float32: there are few of them, and they scale everything else.

**4. The tokenizer.** The token list, the BPE merges, and which tokens are control tokens. It is
tagged `gpt2` with the `gpt-2` pre-tokenizer (lesson 1), so llama.cpp splits text exactly like
`tokenizers` does.

**5. The chat template.** `CHAT_TEMPLATE` is the format of lesson 3 written in Jinja, the template
language llama.cpp and LM Studio use. A message ending in `/think` opens the think span. Any other
message gets `/no_think`.

## Run it

```bash
uv run --extra export python -m theog.export --ckpt runs/theog-50m/best.pt \
    --tokenizer data/tokenizer.json --out release/theog-50m-Q8_0.gguf --quant Q8_0
```

It prints the file's sha256 and writes two Ollama Modelfiles next to it.

## Ollama

Ollama templates are written in Go's template language, which can't check how a string ends. So
there is one Modelfile per mode, each fixing the mode, the stop tokens and theOG's sampling settings
(temperature 0.2, top-k 40, no repeat penalty, context 2048):

```bash
ollama create theog:no_think -f release/Modelfile.no_think
```

```bash
ollama run theog:no_think "Write a function that reverses a list."
```

Without these settings Ollama would use its own defaults (temperature 0.8, for example), and the
model would behave differently from your evaluation.

## Check that the GGUF is the same model

An export can be wrong without any error: a swapped tensor, or a tokenizer that splits text
differently. Check three things with llama.cpp's `llama-server`:

1. **Tokens.** `/tokenize` on a prompt gives the same ids as `tok.encode()`.
2. **Template.** `/apply-template` on a chat gives the exact prompt string of lesson 3, in both modes.
3. **Output.** With greedy decoding (temperature 0), the GGUF and the PyTorch model write the same
   tokens. Use the F16 file for this. Q8_0 rounds the weights, so it may pick a different token
   where two tokens score almost the same.

This course's exporter was checked this way on a small model: all three matched.

## Check it

```bash
uv run --extra export pytest tests/test_08_export.py
```

The test exports the tiny model, reads the file back with `gguf.GGUFReader`, and checks that every
weight is there, that the embedding values survived, and that the architecture is `qwen3`.

## Try this

- Export F16 and Q8_0 and compare the file sizes.
- Load the GGUF in LM Studio and write a `/think` prompt. LM Studio shows control tokens, so you'll
  see the span markers.
