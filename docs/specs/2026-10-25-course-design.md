# DrunkkInference course design (2026-10-25)

## Intent

A course that teaches how to build theOG, a 50M Python code model trained from scratch, step by step:
tokenizer, data, model, training, evaluation, export to GGUF/Ollama.

## Decisions (user)

- Audience: Python developers new to ML.
- Language: English.
- Training: the full-size theOG run on the learner's own GPU (not rented).
- Licence: MIT for code, CC BY 4.0 for lesson text.
- Repo: DrunkkToys/DrunkkInference, private for now.

## Decisions (builder)

- One small package `theog/`, one module per lesson, plus a test per module that runs on CPU.
- Model: exactly the exported theOG-50M v2 (stock qwen3 layout, 50,148,352 params). Research-only
  pieces (Focus bias, depth router) left out.
- Recipe: theOG v2's (AdamW 3e-4, 0.9/0.95, wd 0.01, 5% warmup, cosine to 10%, clip 1.0,
  batch 4 x 8 x 2048, fp32, one epoch).
- Data: only two redistributable public sources, at pinned revisions: nvidia/OpenCodeInstruct
  (CC BY 4.0, rows with all tests passing) and opc-sft-stage2 educational_instruct (MIT). theOG's
  scraped and private sources are excluded.
- Tokenizer: fresh byte-level BPE, 16,000 tokens, special tokens at ids 0-8.
- Eval: HumanEval-164 and MBPP-500 at pinned revisions, subprocess runner, CodeGen-350M-mono as the
  harness sanity gate. graded-30 is not public and is cited only from the model card.
- /think: the user's own pause design; empty span in training; pause-tokens paper cited only as
  separate similar work.

## Lessons

0 what we build, 1 tokenizer, 2 data + contamination, 3 format + packing, 4 model, 5 training,
6 /think span, 7 evaluation, 8 export, 9 publishing.

## Known differences from theOG

- Data mix and size (theOG v2: 24 source files, 833M tokens).
- Token ids (own tokenizer).
- MBPP harness runs the solution before the task setup; theOG's runs setup first.
