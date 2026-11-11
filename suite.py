"""The test suite: which tests, on which items, with which settings.

Every model answers exactly the same items, so two runs can be compared item by item. Change an entry only
with a new SUITE_VERSION; results from different versions are not comparable.
"""

SUITE_VERSION = "2026.11-1"
SEED = 2026  # item selection (--sample-shuffle) and sampling seed

# Tier 2: the capability baseline, once per model on its fastest setup (about one night).
# limit + shuffle = a fixed random subset (the same SEED picks the same items on every run).
TIER2 = {
    # knowledge and reasoning
    "gpqa": dict(task="inspect_evals/gpqa_diamond", epochs=1),                       # 198
    "mmlu_pro": dict(task="inspect_evals/mmlu_pro", limit=700, shuffle=True),        # 700 of 12,032
    # math
    "aime2025": dict(task="inspect_evals/aime2025", epochs=4),                       # 30 x 4
    # instruction following
    "ifeval": dict(task="inspect_evals/ifeval"),                                     # 541
    # coding (rule-scored, run in Python)
    "livebench_coding": dict(task="inspect_evals/livebench", args={"category": "coding"}),
    # tool use
    "bfcl_multi_turn": dict(task="inspect_evals/bfcl", args={"categories": "multi_turn_base"}, limit=100, shuffle=True),
    "bfcl_live": dict(task="inspect_evals/bfcl", args={"categories": "live_multiple"}, limit=200, shuffle=True),
    # long context (tasks/ruler.py; rule-scored). 5 tasks x 8 items at each length
    "ruler": dict(task="tasks/ruler.py", args={"lengths": "32768,131072", "per_task": 8}),
}
# Run by their own harnesses (see README): terminal_bench (Terminal-Bench 2 via Harbor, run_tb.py) and
# cybench (our CTF harness, Codex, 40 x 3).

TIERS = {"2": TIER2}
