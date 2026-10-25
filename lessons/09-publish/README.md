# Lesson 9: publishing

A model is useful to others when they can download it, run it with one command, and trust its
numbers. This lesson is a checklist, not code.

## 1. A model card people can check

The `README.md` of a Hugging Face model repo is its **model card**. Put in it:

- **What it does and its limits.** For example: Python function tasks, 2048 tokens of context.
- **Scores, each with its source.** For every number, give the results file from lesson 7 and the
  settings: file (Q8_0 or F16), mode, temperature, top-k. Give the same numbers for a known baseline
  on the same harness, so readers can compare.
- **How to run it.** The `ollama run` command, and the sampling settings.
- **Training data and its licences** (step 3).
- **Hashes.** The sha256 the export printed, so people can check their download.

theOG's card is an example: [huggingface.co/DrunkkToys/theOG-50M](https://huggingface.co/DrunkkToys/theOG-50M).

## 2. Upload

```bash
uv run --with "huggingface_hub[cli]" hf auth login
```

```bash
uv run --with "huggingface_hub[cli]" hf upload <your-name>/<model-name> release/ .
```

Ollama can run a GGUF straight from Hugging Face (`ollama run hf.co/<your-name>/<model-name>:Q8_0`).
It reads the prompt template and sampling settings from the `template` and `params` files at the
root of the repo. theOG's repo has both, to copy and adapt.

## 3. Licences

Your model is trained on other people's data, and their licences come with it:

| source | licence | what it asks |
|---|---|---|
| nvidia/OpenCodeInstruct | CC BY 4.0 | credit the source |
| OpenCoder-LLM/opc-sft-stage2 | MIT | keep the copyright notice |

Name both sources in your card, with links. If you add sources, check each licence before you
publish.

## 4. Before you publish, check

- [ ] The repo holds no tokens, keys or private data. Search the files before uploading.
- [ ] The contamination check (lesson 2) ran on the exact data you trained on.
- [ ] Every score in the card comes from a results file you can point to.
- [ ] `ollama run` works from a clean machine with only the published files.
