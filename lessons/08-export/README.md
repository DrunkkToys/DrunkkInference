# Lesson 8: export to GGUF and Ollama

**You write:** the mapping from your weight names to llama.cpp's, and the Ollama Modelfile.
**You get:** a `.gguf` file and `ollama run` on your own model, no Python needed. **Time:** about
30 minutes, plus installing [Ollama](https://ollama.com).

A `.pt` checkpoint needs PyTorch and this repo. A **GGUF** file holds everything a runtime needs in
one file: the weights, the tokenizer, the chat template and the model's shape. llama.cpp, Ollama and
LM Studio load it directly, on a laptop, with no Python.

## What goes into the file: `theog/export.py`

`export_gguf()` is written for you. It writes, in order:

1. **The architecture name and shape.** The file says `qwen3` (lesson 4 explains why) and records
   the numbers a runtime needs to rebuild the model: blocks, hidden size, heads, ffn, context,
   RoPE base, norm eps. Nothing else describes the layout: get one wrong and llama.cpp builds a
   different model around your weights.
2. **The tokenizer.** The token list, the BPE merges, and which tokens are control tokens. It is
   tagged `gpt2` with the `gpt-2` pre-tokenizer (lesson 1), so llama.cpp splits text exactly like
   `tokenizers` does.
3. **The chat template.** `CHAT_TEMPLATE` is the format of lesson 3 in Jinja, the template language
   llama.cpp and LM Studio use. A message ending in `/think` opens the span; any other gets
   `/no_think`.
4. **The weights**, under the names llama.cpp expects. That's your part.

## Exercise 8.1: `tensors`

`PER_BLOCK` at the top of the file maps each weight name inside one of your blocks to its GGUF
name:

```
blocks.3.attn.q.weight   ->  blk.3.attn_q.weight
blocks.3.mlp.down.weight ->  blk.3.ffn_down.weight
```

Return every weight as `(gguf_name, float32 numpy array)`, in this order:

1. `token_embd.weight`: your `embed.weight`. There is no separate output weight: `qwen3` with no
   `output.weight` in the file uses the embedding as the output head, which is exactly your tied
   head (lesson 4, step 8).
2. For each block `i`, every `PER_BLOCK` entry, as `blk.{i}.{gguf name}`.
3. `output_norm.weight`: your final `norm.weight`.

Then `assert len(out) == len(model.state_dict())`. A weight with no slot would otherwise be silently
left out of the file, and you'd find out from a model that writes nonsense, with no error.

```bash
uv run pytest tests/test_08_export.py -k tensor_names
```

<details><summary>Solution</summary>

```python
def tensors(model):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out = [("token_embd.weight", sd["embed.weight"])]
    for i in range(model.cfg.layers):
        out += [(f"blk.{i}.{dst}", sd[f"blocks.{i}.{src}"]) for src, dst in PER_BLOCK.items()]
    out.append(("output_norm.weight", sd["norm.weight"]))
    assert len(out) == len(sd), "a weight has no GGUF slot"
    return out
```

</details>

## Quantization

`F16` stores each weight in 16 bits instead of 32. `Q8_0` stores blocks of 32 weights as 8-bit
integers plus one 16-bit scale per block: 8.5 bits per weight, about half of F16. theOG ships
Q8_0. The norm weights stay float32: there are few of them, and every other number passes through
them.

## Exercise 8.2: `modelfile`

Ollama doesn't use the Jinja template. Its own template language (Go's) can't check how a string
ends, so it can't pick the mode from the message. There is one Modelfile per mode instead, and
`go_template(mode)` (written for you) gives its template.

Return the Modelfile's text, one line each:

```
FROM ./theog.gguf
TEMPLATE """<the go_template(mode) text>"""
PARAMETER stop "<|/Answer|>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|User|>"
PARAMETER temperature 0.2
PARAMETER top_k 40
...
```

The stops come from `STOPS` and the parameters from `SAMPLING`, both at the top of the file. End
with one newline.

Without the `PARAMETER` lines Ollama uses its own defaults (temperature 0.8, a repeat penalty), and
your model would behave differently from your evaluation in lesson 7.

```bash
uv run pytest tests/test_08_export.py
```

All 4 should pass.

<details><summary>Solution</summary>

```python
def modelfile(gguf_path, mode="no_think"):
    lines = [f"FROM {gguf_path}", f'TEMPLATE """{go_template(mode)}"""']
    lines += [f'PARAMETER stop "{s}"' for s in STOPS]
    lines += [f"PARAMETER {k} {v}" for k, v in SAMPLING.items()]
    return "\n".join(lines) + "\n"
```

</details>

## Run it

```bash
uv run --extra export python -m theog.export --ckpt runs/laptop/best.pt \
    --tokenizer data/tokenizer.json --out release/laptop-Q8_0.gguf --quant Q8_0
```

```text
release/laptop-Q8_0.gguf  Q8_0  sha256 3852c61732174e7a0ac598c573816be7c36a9b4c1118dcac907076ada195a418
release/Modelfile.no_think  ->  ollama create theog:no_think -f release/Modelfile.no_think
release/Modelfile.think  ->  ollama create theog:think -f release/Modelfile.think
```

A few seconds. A 9.1 MB file: that's your whole model. (Your hash will differ; it's the hash of
your weights.) The export also sets `num_ctx` in each Modelfile to the model's own context, 512
here, so Ollama never feeds it a window longer than it trained on.

Now in Ollama:

```bash
ollama create mymodel -f release/Modelfile.no_think
```

```bash
ollama run mymodel "Write a function that checks if a string is a palindrome."
```

````text
```python
def is_palindrome(s):
    return s == s[::-1]

def is_palindrome(s):
    return is_palindrome(s)
```
````

That's the answer lesson 5 got from PyTorch, flaw and all: a correct function, then a second one
that calls itself forever. Good: the file is your model, flaws included. Exporting doesn't
fix it and doesn't add to it.

No Python, no repo: this runs anywhere Ollama runs. Delete it with `ollama rm mymodel`.

## Check that the GGUF is the same model

An export can be wrong without any error: a swapped tensor, or a tokenizer that splits text
differently. Before you trust it, check three things (with llama.cpp's `llama-server`, or
Ollama's API):

1. **Tokens.** `/tokenize` on a prompt gives the same ids as `tok.encode()`.
2. **Template.** `/apply-template` on a chat gives the exact prompt string of lesson 3, in both modes.
3. **Output.** With greedy decoding (temperature 0), the GGUF and the PyTorch model write the same
   tokens. Use the F16 file for this. Q8_0 rounds the weights, so it may pick a different token
   where two tokens score almost the same.

On the laptop model we compared five prompts at temperature 0, through Ollama's
`/api/generate` with `"options": {"temperature": 0}`, against `generate(..., temperature=0)` in
PyTorch. Both the F16 and the Q8_0 file wrote exactly the same text on all five.

## theOG's own export

theOG-50M v3's research model has two pieces `qwen3` has no slot for. They are removed at export,
and the scores on the [model card](https://huggingface.co/DrunkkToys/theOG-50M) were measured on the
exported file. The course model never had them, so for you `tensors()` covers everything.

## Going further

1. Export with `--quant F16` too. Compare the sizes. Where does the rest of the Q8_0 file go, if the
   weights are about half?
2. Swap two entries in `PER_BLOCK` (say `attn_q` and `attn_k`) and export. Does anything complain?
   What does `ollama run` write?
3. Load the GGUF in LM Studio and send a message ending in `/think`. What do you see?

<details><summary>Answers</summary>

1. Laptop model: F16 is 17.0 MB, Q8_0 is 9.1 MB. The weights alone are 8,343,552 × 2 bytes =
   16.7 MB in F16 and 8,343,552 × 8.5 / 8 = 8.9 MB in Q8_0. The other 0.2 to 0.3 MB is the
   tokenizer (8,000 tokens and their merges), the chat template, and the float32 norm weights.
2. Nothing complains. q and k have the same shape (256 × 256), so the file is perfectly valid. It's
   a different model, though: every score becomes "my key against your query" instead of the other
   way round. Asked for a palindrome check, at temperature 0, ours wrote a row of underscores and
   zeros, then quote marks until Ollama stopped it ("token repeat limit reached"). That's why the greedy comparison above is worth doing:
   it's the only check that catches this.
3. LM Studio uses the Jinja `CHAT_TEMPLATE` from the file, so a message ending in `/think` opens
   the span. If it shows control tokens, you should see `<|/Think|>` at the start of the reply: the
   model closing the span itself. A model trained only on `/no_think` rows follows it with
   `<|Answer|>` only about one time in three (lesson 6, step 1).

</details>

Next: [lesson 9, publishing](../09-publish/README.md).
