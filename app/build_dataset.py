"""Package the task set (and the run logs) for a Hugging Face dataset.

    python -m app.build_dataset
    hf upload VatsalRoy/mini-agent-state-bench build/hf_dataset . --repo-type dataset

Writes build/hf_dataset/:
  README.md               the dataset card
  data/tasks.jsonl        one row per task (shown in the dataset viewer)
  raw/tasks.yaml          the original task file
  raw/policy.md           the policy the agent must follow
  raw/runs/*.jsonl        every trajectory from the experiments, if present
  raw/results/*.csv       the result tables
"""

import argparse
import json
import shutil
from pathlib import Path

from eval.checker import load_tasks

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/VatsalRoy/agent-reliability-bench"

CARD = """---
license: mit
pretty_name: Mini Agent State Bench
language:
- en
tags:
- agents
- tool-use
- evaluation
- llm-as-judge
- reliability
size_categories:
- n<1K
configs:
- config_name: default
  data_files: data/tasks.jsonl
---

# Mini Agent State Bench

{n} customer-support tasks for testing tool-using LLM agents, graded by **what changes in a database** rather than by what the agent says. Built for [agent-reliability-bench]({repo}).

Each task gives the agent a customer message. The agent works through 8 tools (`get_order`, `issue_refund`, `send_reply`, ...) on a small SQLite shop with a one-page refund policy. A run passes only if every **required effect** happened and no **forbidden effect** did. About half the tasks are traps where the correct action is to refuse, or to put a ticket on hold instead of refunding.

## Fields

| Field | Meaning |
|---|---|
| `id` | task name |
| `difficulty` | easy, medium, or hard |
| `category` | lookup, refund, cancel, policy_no, ticket, multi_step |
| `user_message` | what the customer says |
| `required_effects` | database changes that must happen (JSON string) |
| `forbidden_effects` | database changes that must not happen (JSON string) |
| `reference_solution` | one correct sequence of tool calls (JSON string) |

Effects use this format: `table` and `action` (insert, update, delete), plus `match` (for new or deleted rows), or `where` and `set` (for updates), and an optional exact `count`. Values can be plain or `{{"contains": ...}}`, `{{"contains_any": [...]}}`, `{{"in": [...]}}`.

To run the tasks you need the shop itself (database seed, tools, and checker), which lives in the [GitHub repo]({repo}). `raw/policy.md` is the policy text the agent sees.

## Run logs

`raw/runs/` holds every trajectory from the experiments in the repo: {runs}. Each line is one attempt with the full conversation, every tool call and result, the database diff, and the checker's verdict. `raw/results/` has the summary tables.

## Headline results (qwen3.5:4b, 5 runs per task)

The agent claimed success on 98% of attempts; the database confirmed 65% (pass@1). Only 29% of tasks succeeded all 5 times (pass^5). An LLM judge approved 60% of the runs that left the database wrong. See the repo's README and REPORT for method and limitations.

## Limitations

Small (17 tasks), synthetic, and written by one person. Correctness is defined by the task specs. Treat results as directional.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "hf_dataset"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "raw" / "runs").mkdir(parents=True)
    (out / "raw" / "results").mkdir(parents=True)

    tasks = load_tasks()
    with open(out / "data" / "tasks.jsonl", "w") as f:
        for t in tasks:
            # Effects have mixed value types, so they're stored as JSON strings
            # to keep every column a single type for the dataset viewer.
            f.write(json.dumps({
                "id": t["id"],
                "difficulty": t.get("difficulty"),
                "category": t.get("category"),
                "user_message": t["user_message"],
                "required_effects": json.dumps(t["required_effects"]),
                "forbidden_effects": json.dumps(t.get("forbidden_effects", [])),
                "reference_solution": json.dumps(t["reference_solution"]),
            }) + "\n")

    for src in sorted((ROOT / "tasks").glob("*.yaml")):
        shutil.copy(src, out / "raw" / src.name)
    shutil.copy(ROOT / "env" / "policy.md", out / "raw" / "policy.md")
    copied = []
    for src in sorted((ROOT / "runs").glob("*.jsonl")):
        if src.name in ("errors.jsonl", "try.jsonl"):
            continue
        shutil.copy(src, out / "raw" / "runs" / src.name)
        copied.append(src.stem)
    for src in sorted((ROOT / "results").glob("*.csv")):
        shutil.copy(src, out / "raw" / "results" / src.name)

    runs = ", ".join(f"`{c}`" for c in copied) or "none yet"
    (out / "README.md").write_text(CARD.format(n=len(tasks), repo=REPO_URL, runs=runs))
    print(f"Wrote {out} ({len(tasks)} tasks, {len(copied)} run logs)")


if __name__ == "__main__":
    main()
