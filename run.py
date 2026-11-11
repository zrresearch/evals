#!/usr/bin/env python3
"""Run a tier of the suite against one served model (an OpenAI-compatible endpoint).

  run.py --label ripple-65b-4x --model Ripple-65B-Preview --base-url http://<head>:8888/v1 \
         [--tier 2] [--only gpqa,ifeval] [--conc 4] [--temperature 0.6] [--top-p 0.95] [--max-tokens 32768] \
         [--note "build, pack, Sparks, streams"] [--quick]

Inspect runs as the unprivileged EVAL_USER from its venv; raw logs stay in that account's home
(EVAL_HOME/logs/<label>/<test>/). Per-sample scores go to results/<label>/<test>.jsonl and the run's settings to
results/<label>/manifest.json. The endpoint is never written to results. A test already summarized is skipped,
so a stopped run picks up where it left off. --quick: 5 items per test, a wiring check (results/quick-<label>).
"""
import argparse, datetime, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

from suite import SEED, SUITE_VERSION, TIERS

EVAL_USER = os.environ.get("EVAL_USER", "")  # the unprivileged account Inspect runs as (set it in the environment)
EVAL_HOME = Path(os.environ.get("EVAL_HOME", f"/home/{EVAL_USER}/evals"))
REPO = Path(__file__).resolve().parent


def as_user(cmd, env):
    e = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    full = f"cd {EVAL_HOME} && env {e} {cmd}"
    if os.geteuid() == 0:
        return subprocess.run(["runuser", "-u", EVAL_USER, "--", "bash", "-c", full], capture_output=True, text=True)
    return subprocess.run(["bash", "-c", full], capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--tier", default="2")
    p.add_argument("--only")
    p.add_argument("--conc", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--timeout", type=int, default=3600, help="seconds per request")
    p.add_argument("--note", default="")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if not EVAL_USER:
        sys.exit("set EVAL_USER (the unprivileged account that owns the Inspect venv)")

    suite = TIERS[a.tier]
    names = a.only.split(",") if a.only else list(suite)
    label = f"quick-{a.label}" if a.quick else a.label
    out = REPO / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {
        "label": label, "model": a.model, "suite": SUITE_VERSION, "tier": a.tier, "seed": SEED, "note": a.note,
        "sampling": {"temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens},
        "concurrency": a.conc, "tests": {}}
    if man["suite"] != SUITE_VERSION:
        sys.exit(f"{label} was run with suite {man['suite']}; this is {SUITE_VERSION}. Use a new label.")
    env = {"SPARK_BASE_URL": a.base_url, "SPARK_API_KEY": os.environ.get("SPARK_API_KEY", "none"),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "INSPECT_LOG_FORMAT": "eval"}

    # the eval account can't read this repo: copy what it runs into its folder
    shutil.copy(REPO / "summarize.py", EVAL_HOME / "summarize.py")
    shutil.copytree(REPO / "tasks", EVAL_HOME / "tasks", dirs_exist_ok=True)
    if os.geteuid() == 0:
        for f in [EVAL_HOME / "summarize.py", EVAL_HOME / "tasks", *(EVAL_HOME / "tasks").iterdir()]:
            shutil.chown(f, EVAL_USER, EVAL_USER)
    for name in names:
        dst = out / f"{name}.jsonl"
        if dst.exists():
            print(f"{name}: done already, skipping"); continue
        t = suite[name]
        logdir = EVAL_HOME / "logs" / label / name
        cmd = [".venv/bin/inspect", "eval", t["task"], "--model", f"openai-api/spark/{a.model}",
               "--log-dir", str(logdir), "--display", "none", "--no-fail-on-error",
               "--max-connections", str(a.conc), "--temperature", str(a.temperature), "--top-p", str(a.top_p),
               "--max-tokens", str(a.max_tokens), "--seed", str(SEED), "--timeout", str(a.timeout)]
        for k, v in t.get("args", {}).items():
            cmd += ["-T", f"{k}={v}"]
        if "epochs" in t:
            cmd += ["--epochs", str(t["epochs"])]
        if t.get("shuffle"):
            cmd += ["--sample-shuffle", str(SEED)]
        lim = 5 if a.quick else t.get("limit")
        if lim:
            cmd += ["--limit", str(lim)]
        start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        print(f"{name}: running ({start})", flush=True)
        r = as_user(" ".join(shlex.quote(c) for c in cmd), env)
        logs = sorted(logdir.glob("*.eval"), key=lambda f: f.stat().st_mtime)
        if r.returncode != 0 or not logs:
            print(f"{name}: FAILED\n{(r.stderr or r.stdout)[-2000:]}", flush=True)
            man["tests"][name] = {"status": "failed", "started": start}
            mf.write_text(json.dumps(man, indent=1)); continue
        s = as_user(f".venv/bin/python summarize.py {shlex.quote(str(logs[-1]))}", {})
        if s.returncode != 0:
            print(f"{name}: could not read the log\n{s.stderr[-2000:]}", flush=True); continue
        lines = s.stdout.strip().splitlines()
        run = json.loads(lines[-1])["_run"]
        dst.write_text("\n".join(lines[:-1]) + "\n")
        man["tests"][name] = {"status": run["status"], "task": run["task"], "task_args": run["task_args"],
                              "metrics": run["metrics"], "started": run["started"], "completed": run["completed"],
                              "inspect": run["inspect"], "log": logs[-1].name}
        mf.write_text(json.dumps(man, indent=1))
        rows = [json.loads(l) for l in lines[:-1]]
        sc = [x["score"] for x in rows if x["score"] is not None]
        print(f"{name}: {run['status']}, {len(rows)} samples, mean {sum(sc) / max(len(sc), 1):.3f}, "
              f"{sum(x['output_tokens'] or 0 for x in rows):,} output tokens", flush=True)


if __name__ == "__main__":
    main()
