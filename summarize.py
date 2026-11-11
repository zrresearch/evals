#!/usr/bin/env python3
"""One Inspect log -> one line per sample (JSON on stdout): id, epoch, score in [0, 1], tokens.

  summarize.py <log.eval>

Runs inside the Inspect venv. Keeps no prompts, replies or URLs, only what the comparison needs.
"""
import json, sys
from inspect_ai.log import read_eval_log

# scores that are a dict of values: the one we report
DICT_KEY = {"ifeval": "prompt_level_strict"}


def value(v, task):
    if isinstance(v, dict):
        for k in (DICT_KEY.get(task), "accuracy", "score"):
            if k and k in v:
                return value(v[k], task)
        return None
    if v in ("C", True):
        return 1.0
    if v in ("I", False, "N"):
        return 0.0
    if v == "P":
        return 0.5
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(path):
    log = read_eval_log(path)
    task = log.eval.task.split("/")[-1]
    key = "ifeval" if "ifeval" in task else task
    for s in log.samples or []:
        sc = next(iter((s.scores or {}).values()), None)
        u = next(iter((s.model_usage or {}).values()), None)
        print(json.dumps({
            "id": s.id, "epoch": s.epoch,
            "score": value(sc.value, key) if sc else None,
            "error": bool(s.error),
            "output_tokens": u.output_tokens if u else None,
            "input_tokens": u.input_tokens if u else None,
            "seconds": round(s.total_time, 1) if s.total_time else None,
            "limit": s.limit.type if s.limit else None,
        }))
    print(json.dumps({"_run": {
        "task": log.eval.task, "status": log.status, "model": log.eval.model.split("/")[-1],
        "inspect": log.eval.packages, "task_args": log.eval.task_args, "config": log.plan.config.model_dump(exclude_none=True),
        "metrics": {n: {m: x.value for m, x in sc.metrics.items()} for sc in (log.results.scores if log.results else []) for n in [sc.name]},
        "started": log.stats.started_at, "completed": log.stats.completed_at,
    }}))


if __name__ == "__main__":
    main(sys.argv[1])
