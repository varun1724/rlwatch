# Command line

The `rlwatch` CLI is installed automatically with the package. Four subcommands.

## `rlwatch init`

Generate a starter `rlwatch.yaml` in the current directory with every default value populated. Edit it to taste.

```bash
rlwatch init
```

If `rlwatch.yaml` already exists, you're prompted before overwriting.

## `rlwatch runs`

List every monitored run in the local SQLite store.

```bash
rlwatch runs
rlwatch runs --log-dir ./other-logs
```

Output:

```
                              Training Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Run ID                        ┃ Framework  ┃ Started At          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ grpo_v3_exp12                 │ manual     │ 2026-04-11 09:32:14 │
│ run_20260411_083021_aa97c3    │ manual     │ 2026-04-11 08:30:21 │
└───────────────────────────────┴────────────┴─────────────────────┘
```

## `rlwatch diagnose`

Retrospective report on a completed run. By default it picks the latest run; pass `--run-id` to target a specific one.

```bash
rlwatch diagnose
rlwatch diagnose --run-id grpo_v3_exp12
rlwatch diagnose --format json     # for piping into other tools
```

The rich-formatted output gives you:

- **Run header** — id, framework, start time, total steps, overall health (`healthy` / `warning` / `critical`)
- **Metric summaries** — min/max/mean/first/last/trend for entropy, KL, reward, advantage std, loss, gradient norm
- **Alert table** — every alert that fired during the run with detector, severity, message, and recommendation

The JSON output has the same content as a structured dict, ready for `jq` or downstream tooling.

## `rlwatch dashboard`

Launch the Streamlit dashboard at `http://localhost:8501`.

```bash
rlwatch dashboard
rlwatch dashboard --port 8502 --host 127.0.0.1
```

The dashboard requires the `[dashboard]` extra:

```bash
pip install "rlwatch[dashboard]"
```

If you forgot the extra, the command prints a friendly install hint and exits non-zero — it doesn't crash.

The dashboard pulls data from the same SQLite file the CLI reads. You can run it concurrently with an active training process; WAL mode keeps reads and writes from blocking each other.
