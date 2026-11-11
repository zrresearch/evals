# evals

The standard tests for models served on the fleet, so that any two models (or two builds of one model) can be
compared item by item. The method is in SparkMan's `docs/playbooks/model-eval.md`; this repo holds the suite,
the runners and the results.

## Tiers

| Tier | When | What | Time |
|---|---|---|---|
| 0. Serving check | every new deployment | plain reply, tool call, golden replies, SparkMan decode + prefill | < 30 min |
| 1. Quick quality | every new model or build | SparkMan quality suite + tool-eval quick; ppl + KL for our own quants | 1–2 h |
| 2. Capability baseline | once per model, on its fastest setup | the suite below | one night |
| 3. Deep | a specific question | full NYU CTF, all three CTF harnesses, SWE-bench Verified | days |

## Tier 2 (suite 2026.11-1)

| Area | Test | Items | Scored by |
|---|---|---|---|
| Knowledge, reasoning | GPQA Diamond | 198 | letter match |
| | MMLU-Pro, fixed random subset | 700 | letter match |
| Math | AIME 2025 | 30 × 4 samples | exact answer |
| Instruction following | IFEval | 541 | rule checks (prompt-level, strict) |
| Coding | LiveBench coding | all | unit tests |
| Tool use | BFCL multi-turn base | 100 | state and call match |
| | BFCL live multiple | 200 | call match |
| Long context | RULER-style, 32K and 128K (`tasks/ruler.py`) | 5 tasks × 8 × 2 | string match |
| Agentic coding | Terminal-Bench 2, Terminus 2 agent (`run_tb.py`) | 89 | task tests |
| Security | Cybench, Codex harness (the CTF harness in the apex repo) | 40 × 3 | flag |

Nothing is graded by a model. Item sets and the seed are fixed in `suite.py`; changing them means a new suite
version, and runs from different versions are not compared.

Defaults: temperature 0.6, top-p 0.95, 32K max output tokens, 4 concurrent requests, unless the model's card
says otherwise (record it in `--note`). Serve the model with at least as many streams as `--conc`.

## Running

Inspect AI, inspect_evals and Harbor live in Python environments in the eval account's home (`~/evals/.venv`,
`~/evals/.venv-tb`, made with uv); the runners call them as that account, named by `EVAL_USER` in the environment. Install:

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python inspect-ai openai \
  "inspect_evals[livebench,ifeval] @ git+https://github.com/UKGovernmentBEIS/inspect_evals@50b8b19c3" \
  "livebench @ git+https://github.com/LiveBench/LiveBench.git"
uv venv --python 3.12 .venv-tb && uv pip install --python .venv-tb/bin/python harbor
```

```
python3 run.py    --label <model>-<setup> --model <served name> --base-url http://<head>:<port>/v1 --note "<build, pack, Sparks, streams>"
python3 run_tb.py --label <model>-<setup> --model <served name> --base-url http://<head>:<port>/v1
python3 compare.py <label A> <label B>
```

`--quick` runs a handful of items per test to check the wiring (results under `quick-<label>`). A stopped run
resumes: tests already summarized are skipped.

`results/<label>/` holds one line per item per test (id, score, tokens, time) and `manifest.json` (settings,
versions, per-test metrics). Raw Inspect logs, with prompts and replies, stay in the eval account's home. No
endpoint, hostname or address is written to results.

## Comparing

`compare.py` matches items by id and reports, per test: both means, the difference with a 95% bootstrap interval
over items, an exact sign test over items where the two differ, and median output tokens per item. Read the
token column next to the score: a model that needs twice the tokens for the same score takes twice as long on
the same hardware.

## Known limits

- Every public test here can be in a model's training data. That inflates absolute scores; it matters less for
  paired comparisons between builds of one model.
- RULER here is a rule-scored reimplementation of five of its tasks, not the original harness.
- LiveBench coding: inspect_evals pins a snapshot whose 128 problems date from mid-2024 (LiveBench retired them in
  April 2025); the suite pins their release date so they load.
