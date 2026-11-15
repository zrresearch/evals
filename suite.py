"""The test suite: which tests, on which items, with which settings.

Every model answers exactly the same items, so two runs can be compared item by item. Change an entry only
with a new SUITE_VERSION; results from different versions are not comparable.
"""

SUITE_VERSION = "2026.11-2"  # -2: MMLU-Pro pinned to a listed 700; AIME samples unseeded
SEED = 2026  # item selection (--sample-shuffle) and sampling seed (except tests with seed=False)

# Tier 2: the capability baseline, once per model on its fastest setup (about one night).
# limit + shuffle = a fixed random subset (the same SEED picks the same items on every run), but only for tasks that
# load their data in a fixed order. ids = a file of item ids, for tasks that do not.
TIER2 = {
    # knowledge and reasoning
    "gpqa": dict(task="inspect_evals/gpqa_diamond", epochs=1),                       # 198
    # the task shuffles its data unseeded by default, so 2026.11-1 drew a different 700 each run; these are the 700
    # the first run (ripple-65b-4x) drew
    "mmlu_pro": dict(task="inspect_evals/mmlu_pro", args={"shuffle": "false"}, ids="sets/mmlu_pro_700.txt"),  # 700 of 12,032
    # math
    # no request seed: with one, the server returned the same reply in all four epochs
    "aime2025": dict(task="inspect_evals/aime2025", epochs=4, seed=False),          # 30 x 4
    # instruction following
    "ifeval": dict(task="inspect_evals/ifeval"),                                     # 541
    # coding (rule-scored, run in Python)
    # inspect_evals' pinned snapshot: 128 problems from 2024 releases, all retired by LiveBench in 2025-04, so pin
    # the release date they were live at or the task loads nothing
    "livebench_coding": dict(task="inspect_evals/livebench", args={"category": "coding", "livebench_release_date": "2024-07-26"}),
    # tool use
    "bfcl_multi_turn": dict(task="inspect_evals/bfcl", args={"categories": "multi_turn_base"}, limit=100, shuffle=True),
    "bfcl_live": dict(task="inspect_evals/bfcl", args={"categories": "live_multiple"}, limit=200, shuffle=True),
    # long context (tasks/ruler.py; rule-scored). 5 tasks x 8 items at each length
    "ruler": dict(task="tasks/ruler.py", args={"lengths": "32768,131072", "per_task": 8}),
}
# Run by their own harnesses (see README): terminal_bench (Terminal-Bench 2 via Harbor, run_tb.py) and
# cybench (our CTF harness, Codex, 40 x 3).

TIERS = {"2": TIER2}
