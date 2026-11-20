# Lesson 9: publishing

**You write:** your model's card. **You get:** a model other people can download, run with one
command, and check. **Time:** about 30 minutes.

There's no code in this lesson. A published model is a promise: "this file does what the card
says". This lesson is about keeping it.

## 1. The model card

The `README.md` of a Hugging Face model repo is its **model card**. Readers decide from it whether
to trust your numbers, so every claim in it should point at something they can check.

### Exercise 9.1: write `release/README.md`

Fill this in for the model you trained. Every line has to come from something you ran:

```markdown
# <model name>

A <N>M-parameter Python code model, trained from scratch with
[DrunkkInference](https://github.com/DrunkkToys/DrunkkInference). It writes one Python function
for a task given in English. Context: <window> tokens.

## Run it

    ollama run hf.co/<you>/<model>:Q8_0 "Write a function that reverses a list."

Sampling: temperature 0.2, top-k 40, no repeat penalty (set by the `params` file).

## Scores

| benchmark | passed | file | mode | temperature |
|---|:-:|---|---|:-:|
| HumanEval | <x>/164 | <results file from lesson 7> | /no_think | <t> |
| MBPP-500  | <y>/500 | <results file from lesson 7> | /no_think | <t> |

Harness: `theog.evaluate`, which passes 164/164 and 498/500 reference solutions.

## Training

<rows> rows, <tokens> tokens (data/packed/meta.json), <epochs> epochs, <steps> steps, <minutes> on <hardware>.
Final validation loss <loss>.

## Data and licences

- nvidia/OpenCodeInstruct, CC BY 4.0
- OpenCoder-LLM/opc-sft-stage2 (educational_instruct), MIT
- Rows matching HumanEval or MBPP (13-gram overlap) were removed before training.

## Limits

<what you saw in lesson 7: what it gets wrong>

## Files

| file | sha256 |
|---|---|
| <name>-Q8_0.gguf | <the hash the export printed> |
```

The **Limits** section is the one people skip, and the one readers trust most. Read your lesson 7
failures and say plainly what the model can't do.

## 2. Ollama from Hugging Face

`ollama run hf.co/<you>/<model>:Q8_0` downloads the GGUF straight from your repo. It reads the prompt
template and sampling settings from two files at the repo root, `template` and `params`. Without
them Ollama falls back to its defaults and your model behaves differently from your card.

Make them from your Modelfile: `template` is the text between the `TEMPLATE """` quotes, and
`params` is the stops and parameters as JSON:

```bash
uv run python -c "
import json
from theog.export import go_template, STOPS, SAMPLING
open('release/template', 'w').write(go_template('no_think'))
json.dump({'stop': STOPS, **SAMPLING, 'num_ctx': 512}, open('release/params', 'w'), indent=2)
"
```

`num_ctx` is your model's context: 512 on the laptop track, 2048 on the full track.

theOG's repo has both, to compare: [huggingface.co/DrunkkToys/theOG-50M](https://huggingface.co/DrunkkToys/theOG-50M).

## 3. Upload

```bash
uv run --with "huggingface_hub[cli]" hf auth login
```

```bash
uv run --with "huggingface_hub[cli]" hf upload <you>/<model> release/ .
```

Only `release/` goes up: the GGUF, the README, `template`, `params`. Not the checkpoints, not the
data.

## 4. Licences

Your model is trained on other people's data, and their licences come with it:

| source | licence | what it asks |
|---|---|---|
| nvidia/OpenCodeInstruct | CC BY 4.0 | credit the source |
| OpenCoder-LLM/opc-sft-stage2 | MIT | keep the copyright notice |

Name both in your card, with links. If you add sources, check each licence before you publish.

## 5. Before you publish

- [ ] No tokens, keys or private data in `release/`. Look at every file.
- [ ] The contamination check (lesson 2) ran on the exact data you trained on.
- [ ] Every score in the card comes from a results file you kept.
- [ ] `ollama run hf.co/<you>/<model>:Q8_0` works on a machine that has never seen this repo.
- [ ] The sha256 in the card is the one of the uploaded file.

## Where to go from here

You've built the whole pipeline once. Everything that makes a model better is a change to one of
its steps, measured in lesson 7:

- **More data** (lesson 2): the full track, or more sources. theOG used 12.
- **A bigger model** (lesson 4): the full track is 50M. Every size you try costs a training run.
- **Better data** (lesson 2): a new cleaning rule, a tighter skeleton cap.
- **The think span** (lesson 6): a design of your own, scored by its total cost in nats, or the
  screen's best design trained for 3 epochs.

Change one thing at a time, and score both benchmarks every time. That's the whole method.

Back to the [start](../../README.md).
