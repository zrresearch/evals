# Notes for AI agents working on this repo

This repository is **public**. Everything committed here, results included, is published the moment it is pushed.

- **No site details:** no hostnames, IP addresses, usernames, home-directory paths or personal details, in code,
  READMEs, manifests, run logs or result files. Use placeholders such as `<head>` and `<eval-host>`, and pass real
  values in through the environment or the command line.
- **No secrets:** read API keys from the environment by name (`SPARK_API_KEY`); never write a key into a file.
- **Before every push**, search what you are adding for hostnames, private addresses (192.168., 10., 100.64-127.,
  172.16-31.), `/home/` paths and key-like strings, including in `*.run.log` and `manifest.json`.
- **Papers and drafts don't go here.** Write-ups are published on zrresearch.org; this repo holds the suite, the
  runners and per-item results.
- **Suite changes** bump `SUITE_VERSION` in `suite.py` and get a note in the README, so results from different
  versions are never compared by mistake.
