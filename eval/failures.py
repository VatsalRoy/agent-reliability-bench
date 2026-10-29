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

Families, to compare with ThinkingBox's "most failures are tool handling":
  decision  the agent used the tools but decided wrong:
            policy_violation, wrong_value
  tool_use  the agent didn't carry out the work through the tools:
            everything else
This split is our own definition, so the comparison with ThinkingBox is loose.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from eval.checker import effect_matches

ORDER = ["step_limit", "no_state_change", "policy_violation", "tool_error_unrecovered",
         "duplicate_effect", "wrong_value", "no_reply_sent", "missing_effect"]
DECISION = {"policy_violation", "wrong_value"}


def family(signature: str) -> str:
    return "decision" if signature in DECISION else "tool_use"


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
    sigs = Counter(classify(r) for r in failed)
    fams = Counter(family(s) for s in sigs.elements())
    by_task = {}
    for r in failed:
        by_task.setdefault(r["task_id"], Counter())[classify(r)] += 1
    return {"failed": len(failed), "total": len(records), "signatures": sigs,
            "families": fams, "by_task": by_task}


def report(t: dict) -> str:
    n = t["failed"] or 1
    lines = [f"{t['failed']} failed attempts out of {t['total']}", "", "SIGNATURES"]
    for sig in ORDER:
        if t["signatures"][sig]:
            lines.append(f"  {sig:<24} {t['signatures'][sig]:>3}  ({t['signatures'][sig] / n:.0%})  [{family(sig)}]")
    lines += ["", "FAMILIES"]
    for fam in ("tool_use", "decision"):
        lines.append(f"  {fam:<24} {t['families'][fam]:>3}  ({t['families'][fam] / n:.0%})")
    lines += ["", "BY TASK"]
    for task, c in sorted(t["by_task"].items()):
        lines.append(f"  {task:<34} " + ", ".join(f"{s} x{k}" for s, k in c.most_common()))
    return "\n".join(lines)


def write_outputs(t: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signature", "family", "count", "share"])
        for sig in ORDER:
            w.writerow([sig, family(sig), t["signatures"][sig],
                        round(t["signatures"][sig] / (t["failed"] or 1), 4)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from eval.metrics import BLUE, GRID, INK, INK_2, ORANGE, SURFACE

    rows = [(s, t["signatures"][s]) for s in ORDER if t["signatures"][s]]
    rows.sort(key=lambda r: r[1])
    colors = {"tool_use": BLUE, "decision": ORANGE}
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([r[0].replace("_", " ") for r in rows], [r[1] for r in rows], height=0.5,
            color=[colors[family(r[0])] for r in rows], zorder=2)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.15, i, str(v), va="center", color=INK, fontsize=10)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=INK, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("failed attempts", color=INK_2)
    fam = t["families"]
    ax.legend(handles=[Patch(color=BLUE, label=f"tool use ({fam['tool_use']})"),
                       Patch(color=ORANGE, label=f"decision ({fam['decision']})")],
              frameon=False, loc="lower right", labelcolor=INK)
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