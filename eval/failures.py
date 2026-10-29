"""Sort failed attempts into failure signatures.

    python -m eval.failures runs/main_k5.jsonl

Each failed attempt gets exactly one signature, checked in this order:

  step_limit              ran out of steps
  no_state_change         changed nothing at all (not even a reply)
  policy_violation        made a change the task forbids (refund the policy
                          says no to, cancelled a shipped order, ...)
  tool_error_unrecovered  a tool returned an error and a required change is missing
  duplicate_effect        did a required thing too many times (e.g. two replies)
  wrong_value             made the right kind of change with the wrong values
                          (wrong amount, reply missing required info, ...)
  no_reply_sent           did the work but never called send_reply
                          (often: answered in chat instead)
  missing_effect          skipped a required change (e.g. left a ticket open)

Each signature is also mapped to ThinkingBox's four failure categories
(Li et al., arXiv 2608.19741), so the two can be compared directly:
  tool_usage               failed to recover from tool errors / no progress
                           (tool_error_unrecovered, step_limit)
  no_state_changing_action omitted a required create/update/cancel/refund
                           (missing_effect; no_state_change when a non-reply
                           change was required)
  incomplete_resolution    the customer wasn't properly answered: no reply,
                           two replies, or a reply missing required info
  wrong_state_update       a change was made but the resulting state is wrong
                           (policy_violation, wrong_value on a non-reply table)
The mapping is our reading of their definitions.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from eval.checker import effect_matches

ORDER = ["step_limit", "no_state_change", "policy_violation", "tool_error_unrecovered",
         "duplicate_effect", "wrong_value", "no_reply_sent", "missing_effect"]
CATEGORIES = ["tool_usage", "no_state_changing_action", "incomplete_resolution", "wrong_state_update"]


def category(record: dict, signature: str) -> str:
    """Map a signature to a ThinkingBox failure category."""
    missing_tables = {m["table"] for m in record["checker"]["missing"]}
    only_reply = missing_tables <= {"replies"}
    if signature in ("tool_error_unrecovered", "step_limit"):
        return "tool_usage"
    if signature == "policy_violation":
        return "wrong_state_update"
    if signature in ("no_reply_sent", "duplicate_effect", "wrong_value", "no_state_change"):
        if only_reply:
            return "incomplete_resolution"
        return "no_state_changing_action" if signature == "no_state_change" else "wrong_state_update"
    return "no_state_changing_action"  # missing_effect


def classify(record: dict) -> str | None:
    if record["verified_success"]:
        return None
    ck = record["checker"]
    if record["finish_reason"] == "max_steps":
        return "step_limit"
    if ck["no_state_change"]:
        return "no_state_change"
    if ck["forbidden_hit"]:
        return "policy_violation"
    if record["tool_errors"] and ck["missing"]:
        return "tool_error_unrecovered"

    found = set()
    for miss in ck["missing"]:
        if miss.get("count") is not None and miss["found"] > miss["count"]:
            found.add("duplicate_effect")
            continue
        same_kind = [c for c in ck["diff"] if c["table"] == miss["table"] and c["action"] == miss["action"]
                     and not effect_matches(miss, c)]
        if same_kind:
            found.add("wrong_value")
        elif miss["table"] == "replies":
            found.add("no_reply_sent")
        else:
            found.add("missing_effect")
    for sig in ORDER:
        if sig in found:
            return sig
    return "missing_effect"


def tally(records: list[dict]) -> dict:
    failed = [r for r in records if not r["verified_success"]]
    sigs, cats, pairs, by_task = Counter(), Counter(), Counter(), {}
    for r in failed:
        sig = classify(r)
        cat = category(r, sig)
        sigs[sig] += 1
        cats[cat] += 1
        pairs[(sig, cat)] += 1
        by_task.setdefault(r["task_id"], Counter())[sig] += 1
    return {"failed": len(failed), "total": len(records), "signatures": sigs,
            "categories": cats, "pairs": pairs, "by_task": by_task}


def report(t: dict) -> str:
    n = t["failed"] or 1
    lines = [f"{t['failed']} failed attempts out of {t['total']}", "", "SIGNATURES"]
    for sig in ORDER:
        if t["signatures"][sig]:
            lines.append(f"  {sig:<24} {t['signatures'][sig]:>3}  ({t['signatures'][sig] / n:.0%})")
    lines += ["", "THINKINGBOX CATEGORIES"]
    for cat in CATEGORIES:
        lines.append(f"  {cat:<26} {t['categories'][cat]:>3}  ({t['categories'][cat] / n:.0%})")
    lines += ["", "BY TASK"]
    for task, c in sorted(t["by_task"].items()):
        lines.append(f"  {task:<34} " + ", ".join(f"{s} x{k}" for s, k in c.most_common()))
    return "\n".join(lines)


def write_outputs(t: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signature", "thinkingbox_category", "count", "share"])
        for (sig, cat), c in sorted(t["pairs"].items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
            w.writerow([sig, cat, c, round(c / (t["failed"] or 1), 4)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from eval.metrics import BLUE, GRID, INK, INK_2, ORANGE, SURFACE

    # Fixed categorical slots, one per ThinkingBox category (never reordered).
    colors = dict(zip(CATEGORIES, [BLUE, ORANGE, "#1baf7a", "#eda100"]))
    rows = sorted(t["pairs"].items(), key=lambda kv: kv[1])
    labels = [f"{sig.replace('_', ' ')}" for (sig, _), _ in rows]
    if len(set(labels)) != len(labels):  # same signature in two categories
        labels = [f"{sig.replace('_', ' ')} ({cat.replace('_', ' ')})" for (sig, cat), _ in rows]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(labels, [c for _, c in rows], height=0.5,
            color=[colors[cat] for (_, cat), _ in rows], zorder=2)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.15, i, str(v), va="center", color=INK, fontsize=10)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=INK, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("failed attempts", color=INK_2)
    cats = t["categories"]
    ax.legend(handles=[Patch(color=colors[c], label=f"{c.replace('_', ' ')} ({cats[c]})") for c in CATEGORIES],
              title="ThinkingBox category", title_fontsize=9, frameon=False, loc="lower right",
              labelcolor=INK, fontsize=9)
    ax.set_title(f"Why the agent failed ({t['failed']} failed attempts)", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    path = out_dir / "figures" / "failures.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    records = [json.loads(line) for line in open(args.runs) if line.strip()]
    t = tally(records)
    print(report(t))
    path = write_outputs(t, Path(args.out))
    print(f"\nWrote {Path(args.out) / 'failures.csv'} and {path}")


if __name__ == "__main__":
    main()
