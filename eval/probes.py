"""TRACE-style probes: is a score change the agent's, or the grader's?

    python -m eval.probes runs/main_k5.jsonl \\
        control=runs/control_k5.jsonl rename=runs/probe_rename_k5.jsonl \\
        reformat=runs/probe_reformat_k5.jsonl misleading=runs/probe_misleading_k5.jsonl

Every run is graded twice:
  state checker   the database diff (our ground truth)
  name grader     a deliberately naive grader that matches the tool calls in
                  the transcript against the reference solution by tool NAME,
                  the way many trajectory-matching benchmarks do

For each probe vs the base run we report:
  delta    change in pass@1, with a 95% CI from a paired bootstrap over tasks
  flips    share of (task, attempt) pairs whose verdict changed

The control is the same setup with different seeds. Its delta should be
about zero and its flips show how much verdicts change from randomness alone.
A probe "has an effect" only if its delta CI excludes zero.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.checker import load_tasks

WRITE_TOOLS = {"create_ticket", "update_ticket_status", "issue_refund", "cancel_order", "send_reply"}
KEY_ARGS = ("order_id", "amount", "status")  # free text and ticket ids are ignored


# ---------- the naive grader ----------
def _key(name: str, args: dict) -> tuple:
    vals = []
    for k in KEY_ARGS:
        if k in args:
            v = args[k]
            try:
                v = round(float(v), 2)
            except (TypeError, ValueError):
                v = str(v).lower()
            vals.append((k, v))
    return (name, tuple(vals))


def name_match_grade(record: dict, task: dict) -> bool:
    """PASS if the successful write calls, by name and key arguments, are exactly
    the reference solution's write calls (order ignored)."""
    expected = sorted(_key(s["tool"], s.get("args", {}))
                      for s in task["reference_solution"] if s["tool"] in WRITE_TOOLS)
    actual = sorted(_key(c["name"], c["args"]) for c in record["tool_calls"]
                    if c["name"] in WRITE_TOOLS and "error" not in c["result"])
    return expected == actual


# ---------- comparing two runs ----------
def _by_key(records):
    return {(r["task_id"], r["attempt"]): r for r in records}


def _paired_delta_ci(base, other, grade, iters=2000, seed=0):
    """Per task: mean(other) - mean(base) over matched attempts. Bootstrap over tasks."""
    per_task = defaultdict(lambda: ([], []))
    for key in base.keys() & other.keys():
        per_task[key[0]][0].append(grade(base[key]))
        per_task[key[0]][1].append(grade(other[key]))
    diffs = [sum(o) / len(o) - sum(b) / len(b) for b, o in per_task.values()]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def compare(base_records, other_records, tasks: dict) -> dict:
    base, other = _by_key(base_records), _by_key(other_records)
    keys = sorted(base.keys() & other.keys())
    if not keys:
        raise ValueError("No matching (task, attempt) pairs between the two runs")
    graders = {
        "state": lambda r: bool(r["verified_success"]),
        "name": lambda r: name_match_grade(r, tasks[r["task_id"]]),
    }
    out = {"pairs": len(keys)}
    for g, grade in graders.items():
        b = sum(grade(base[k]) for k in keys) / len(keys)
        o = sum(grade(other[k]) for k in keys) / len(keys)
        delta, ci = _paired_delta_ci(base, other, grade)
        flips = sum(grade(base[k]) != grade(other[k]) for k in keys) / len(keys)
        out[g] = {"base": b, "other": o, "delta": delta, "ci": ci, "flips": flips,
                  "effect": not (ci[0] <= 0 <= ci[1])}
    return out


# ---------- report ----------
def _fmt_delta(d):
    return f"{d['delta'] * 100:+5.0f} pts [{d['ci'][0] * 100:+.0f}, {d['ci'][1] * 100:+.0f}]"


def report(results: dict) -> str:
    lines = [
        f"{'probe':<12} {'grader':<6} {'base':>5} {'probe':>6}   {'delta (95% CI)':<24} {'flips':>6}  verdict",
        "-" * 82,
    ]
    for name, r in results.items():
        for g in ("state", "name"):
            d = r[g]
            verdict = "EFFECT" if d["effect"] else "within noise"
            lines.append(f"{name:<12} {g:<6} {d['base']:>5.0%} {d['other']:>6.0%}   "
                         f"{_fmt_delta(d):<24} {d['flips']:>6.0%}  {verdict}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "grader", "pairs", "base_pass1", "probe_pass1", "delta",
                    "ci_low", "ci_high", "flip_rate", "effect"])
        for name, r in results.items():
            for g in ("state", "name"):
                d = r[g]
                w.writerow([name, g, r["pairs"], round(d["base"], 4), round(d["other"], 4),
                            round(d["delta"], 4), round(d["ci"][0], 4), round(d["ci"][1], 4),
                            round(d["flips"], 4), d["effect"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.metrics import BLUE, GRID, INK, INK_2, ORANGE, SURFACE

    names = list(results)
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(names) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for offset, g, color, label in ((-0.15, "state", BLUE, "state checker (database)"),
                                    (0.15, "name", ORANGE, "name-matching grader")):
        for i, n in enumerate(names):
            d = results[n][g]
            y = len(names) - 1 - i + offset
            ax.plot([d["ci"][0] * 100, d["ci"][1] * 100], [y, y], color=color, linewidth=2, zorder=2)
            ax.plot(d["delta"] * 100, y, "o", color=color, markersize=7, zorder=3,
                    label=label if i == 0 else None)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("change in pass@1 vs. base run (points, 95% CI)", color=INK_2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              labelcolor=INK, fontsize=9)
    ax.set_title("Did the score change because of the agent or the grader?", loc="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    path = out_dir / "figures" / "probes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="the base run, e.g. runs/main_k5.jsonl")
    ap.add_argument("probes", nargs="+", help="name=path pairs, e.g. rename=runs/probe_rename_k5.jsonl")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    def load(p):
        return [json.loads(line) for line in open(p) if line.strip()]

    tasks = {t["id"]: t for t in load_tasks()}
    base = load(args.base)
    results = {}
    for item in args.probes:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"Expected name=path, got {item!r}")
        results[name] = compare(base, load(path), tasks)
    print(report(results))
    fig = write_outputs(results, Path(args.out))
    print(f"Wrote {Path(args.out) / 'probes.csv'} and {fig}")


if __name__ == "__main__":
    main()