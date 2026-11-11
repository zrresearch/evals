#!/usr/bin/env python3
"""Terminal-Bench 2 through Harbor, with its reference agent (Terminus 2), against one served model.

  run_tb.py --label ripple-65b-4x --model Ripple-65B-Preview --base-url http://<head>:8888/v1 [--conc 4] [--quick]

Each task runs in its own Docker container on this host; --memory limit keeps a runaway task from taking the
host (the CTF lesson). Harbor's job folder stays in the eval account's home (EVAL_HOME/tb-jobs/<label>); one line
per task goes to results/<label>/terminal_bench.jsonl. --quick: 3 tasks, a wiring check.
"""
import argparse, json, os, shlex, subprocess
from pathlib import Path

from run import EVAL_HOME, EVAL_USER, REPO, as_user

DATASET = "terminal-bench@2.0"


def rows(job):
    for f in sorted(job.rglob("result.json")):
        if f.parent == job:
            continue  # the job's own summary
        r = json.loads(f.read_text())
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward", vr.get("reward"))
        ar = r.get("agent_result") or {}
        yield {"id": r.get("task_name") or f.parent.name.rsplit("__", 1)[0], "epoch": 1,
               "score": float(reward) if reward is not None else None,
               "error": bool(r.get("exception_info")),
               "output_tokens": ar.get("n_output_tokens"), "input_tokens": ar.get("n_input_tokens"),
               "seconds": None, "limit": (r.get("exception_info") or {}).get("exception_type")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--conc", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if not EVAL_USER:
        raise SystemExit("set EVAL_USER (the unprivileged account that owns the Harbor venv)")
    label = f"quick-{a.label}" if a.quick else a.label
    out = REPO / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    cmd = [".venv-tb/bin/harbor", "run", "-d", DATASET, "-a", "terminus-2", "-m", f"openai/{a.model}",
           "--base-url", a.base_url, "--api-format", "openai_chat_completions", "--api-key-name", "SPARK_API_KEY",
           "-n", str(a.conc), "-o", str(EVAL_HOME / "tb-jobs"), "--job-name", label, "--memory", "limit", "-y", "-q"]
    if a.quick:
        cmd += ["-l", "3"]
    r = as_user(" ".join(shlex.quote(c) for c in cmd), {"SPARK_API_KEY": os.environ.get("SPARK_API_KEY", "none")})
    if r.returncode != 0:
        print((r.stderr or r.stdout)[-3000:])
    rs = list(rows(EVAL_HOME / "tb-jobs" / label))
    (out / "terminal_bench.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
    sc = [x["score"] for x in rs if x["score"] is not None]
    print(f"terminal_bench: {len(rs)} tasks, {sum(sc):.0f} solved, {sum(x['error'] for x in rs)} errors")


if __name__ == "__main__":
    main()
