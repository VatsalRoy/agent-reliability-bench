"""Reliability metrics from repeated runs.

    python -m eval.metrics runs/main_k5.jsonl

For each task with n attempts and c successes:
  pass@1  = c / n                          "if I try once, how often does it work?"
  pass@k  = 1 - C(n-c, k) / C(n, k)        "does it work at least once in k tries?"
  pass^k  = C(c, k) / C(n, k)              "does it work all k times?"
With k = n, pass^k is simply 1 if every attempt passed, else 0.
Overall numbers are averages over tasks, so every task counts equally.

Confidence intervals (95%):
  - pass@k and pass^k: Wilson interval over tasks.
  - pass@1: bootstrap that resamples whole tasks. Attempts on the same task are
    not independent, so treating 85 attempts as 85 independent coin flips would
    make the interval look falsely narrow.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path


# ---------- estimators ----------
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k)


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


# ---------- grouping ----------
def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def per_task(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[r["task_id"]].append(r)
    rows = []
    for task_id, rs in groups.items():
        n, c = len(rs), sum(r["verified_success"] for r in rs)
        rows.append({
            "task_id": task_id,
            "difficulty": rs[0].get("difficulty"),
            "category": rs[0].get("category"),
            "n": n,
            "c": c,
            "pass@1": c / n,
            "pass@k": pass_at_k(n, c, n),
            "pass^k": pass_hat_k(n, c, n),
            "claimed_rate": sum(r["claimed_success"] for r in rs) / n,
            "mean_seconds": sum(r["seconds"] for r in rs) / n,
            "mean_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs) / n,
        })
    return rows


def summarize(records: list[dict], usd_per_mtok: float = 0.0) -> dict:
    tasks = per_task(records)
    ks = {t["n"] for t in tasks}
    if len(ks) != 1:
        raise ValueError(f"Every task needs the same number of attempts, got {sorted(ks)}")
    k = ks.pop()
    T, N = len(tasks), len(records)

    p1 = [t["pass@1"] for t in tasks]
    pk = sum(t["pass@k"] for t in tasks)
    phk = sum(t["pass^k"] for t in tasks)
    verified = sum(r["verified_success"] for r in records)
    claimed = sum(r["claimed_success"] for r in records)
    explicit = sum(r["outcome_tag"] == "SUCCESS" for r in records)
    false_claims = sum(r["claimed_success"] and not r["verified_success"] for r in records)
    dependable = sum(t["c"] == t["n"] for t in tasks)
    seconds = sum(r["seconds"] for r in records)
    tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)

    def per(x, d):
        return x / d if d else float("inf")

    return {
        "k": k, "tasks": T, "attempts": N,
        "pass@1": sum(p1) / T, "pass@1_ci": bootstrap_mean(p1),
        "pass@k": pk / T, "pass@k_ci": wilson(pk, T),
        "pass^k": phk / T, "pass^k_ci": wilson(phk, T),
        "curve": {j: {"pass@k": sum(pass_at_k(t["n"], t["c"], j) for t in tasks) / T,
                      "pass^k": sum(pass_hat_k(t["n"], t["c"], j) for t in tasks) / T}
                  for j in range(1, k + 1)},
        "claimed_rate": claimed / N,
        "claimed_explicit_rate": explicit / N,
        "claimed_inferred_rate": (claimed - explicit) / N,
        "verified_rate": verified / N,
        "false_claim_rate": false_claims / N,
        "false_claims_among_claims": per(false_claims, claimed),
        "dependable_tasks": dependable,
        "seconds_total": seconds,
        "tokens_total": tokens,
        "seconds_per_success": per(seconds, verified),
        "seconds_per_dependable_task": per(seconds, dependable),
        "tokens_per_success": per(tokens, verified),
        "tokens_per_dependable_task": per(tokens, dependable),
        "usd_per_mtok": usd_per_mtok,
        "usd_per_dependable_task": per(tokens * usd_per_mtok / 1e6, dependable),
        "by_difficulty": _by(tasks, "difficulty"),
        "per_task": tasks,
    }


def _by(tasks, key) -> dict:
    out = defaultdict(list)
    for t in tasks:
        out[t[key]].append(t)
    return {g: {"tasks": len(ts),
                "pass@1": sum(t["pass@1"] for t in ts) / len(ts),
                "pass^k": sum(t["pass^k"] for t in ts) / len(ts)}
            for g, ts in out.items()}


# ---------- output ----------
def pct(x):
    return f"{x:.0%}"


def report(s: dict) -> str:
    k = s["k"]
    lines = [
        f"{s['tasks']} tasks x k={k} = {s['attempts']} attempts",
        "",
        "RELIABILITY (averaged over tasks, 95% CI)",
        f"  pass@1  (works on one try)      {pct(s['pass@1']):>5}   [{pct(s['pass@1_ci'][0])}, {pct(s['pass@1_ci'][1])}]",
        f"  pass@{k}  (works at least once)   {pct(s['pass@k']):>5}   [{pct(s['pass@k_ci'][0])}, {pct(s['pass@k_ci'][1])}]",
        f"  pass^{k}  (works all {k} times)     {pct(s['pass^k']):>5}   [{pct(s['pass^k_ci'][0])}, {pct(s['pass^k_ci'][1])}]",
        f"  Gap pass@1 - pass^{k}:            {(s['pass@1'] - s['pass^k']) * 100:.0f} points",
        "",
        "SAID vs DID (over all attempts)",
        f"  Agent said it succeeded         {pct(s['claimed_rate']):>5}   "
        f"(explicit {pct(s['claimed_explicit_rate'])}, inferred from stopping {pct(s['claimed_inferred_rate'])})",
        f"  Database confirms success       {pct(s['verified_rate']):>5}",
        f"  Said done, DB disagrees         {pct(s['false_claim_rate']):>5}   "
        f"({pct(s['false_claims_among_claims'])} of all success claims were false)",
        "",
        "COST (local model, so measured in time and tokens)",
        f"  Total                           {s['seconds_total'] / 60:.1f} min, {s['tokens_total']:,} tokens",
        f"  Per success                     {s['seconds_per_success']:.1f} s, {s['tokens_per_success']:,.0f} tokens",
        f"  Per dependable task ({s['dependable_tasks']} passed {k}/{k}) "
        f"{s['seconds_per_dependable_task']:.1f} s, {s['tokens_per_dependable_task']:,.0f} tokens",
    ]
    if s["usd_per_mtok"]:
        lines.append(f"  At ${s['usd_per_mtok']}/M tokens: ${s['usd_per_dependable_task']:.4f} per dependable task")
    lines += ["", "BY DIFFICULTY"]
    for g in ("easy", "medium", "hard"):
        if g in s["by_difficulty"]:
            d = s["by_difficulty"][g]
            lines.append(f"  {g:<7} {d['tasks']:>2} tasks   pass@1 {pct(d['pass@1']):>5}   pass^{k} {pct(d['pass^k']):>5}")
    lines += ["", "PER TASK  (successes / attempts)"]
    for t in sorted(s["per_task"], key=lambda t: (-t["pass@1"], t["task_id"])):
        bar = "#" * t["c"] + "." * (t["n"] - t["c"])
        lines.append(f"  {t['task_id']:<34} {bar}  {t['c']}/{t['n']}")
    return "\n".join(lines)


def write_csvs(s: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = s["per_task"]
    with open(out_dir / "per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
        w.writeheader()
        for t in sorted(tasks, key=lambda t: t["task_id"]):
            w.writerow({key: round(v, 4) if isinstance(v, float) else v for key, v in t.items()})
    k = s["k"]
    rows = [
        ("pass@1", s["pass@1"], *s["pass@1_ci"]),
        (f"pass@{k}", s["pass@k"], *s["pass@k_ci"]),
        (f"pass^{k}", s["pass^k"], *s["pass^k_ci"]),
        ("claimed_success_rate", s["claimed_rate"], "", ""),
        ("claimed_explicit_rate", s["claimed_explicit_rate"], "", ""),
        ("verified_success_rate", s["verified_rate"], "", ""),
        ("false_claim_rate", s["false_claim_rate"], "", ""),
        ("dependable_tasks", s["dependable_tasks"], "", ""),
        ("seconds_per_dependable_task", s["seconds_per_dependable_task"], "", ""),
        ("tokens_per_dependable_task", s["tokens_per_dependable_task"], "", ""),
    ]
    with open(out_dir / "main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_low", "ci_high"])
        for name, v, lo, hi in rows:
            w.writerow([name] + [round(x, 4) if isinstance(x, float) else x for x in (v, lo, hi)])


# ---------- figures ----------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def plot(s: dict, model: str, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = s["k"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. The headline: what it said vs what it did, once vs every time.
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    labels = ["Agent said\nit succeeded", "pass@1\n(one try)", f"pass@{k}\n(at least once)", f"pass^{k}\n(all {k} times)"]
    values = [s["claimed_rate"], s["pass@1"], s["pass@k"], s["pass^k"]]
    cis = [None, s["pass@1_ci"], s["pass@k_ci"], s["pass^k_ci"]]
    colors = ["#a8a7a1", BLUE, BLUE, BLUE]
    _style(ax)
    ax.bar(labels, values, width=0.45, color=colors, zorder=2)
    for i, (v, ci) in enumerate(zip(values, cis)):
        if ci:
            ax.plot([i, i], ci, color=INK_2, linewidth=1.2, zorder=3)
        ax.text(i, (ci[1] if ci else v) + 0.03, f"{v:.0%}", ha="center", color=INK, fontsize=11, weight="bold")
    ax.tick_params(axis="x", colors=INK, labelsize=10)
    ax.set_title(f"Said vs. did: {model}, {s['tasks']} tasks x {k} runs", loc="left", color=INK, fontsize=12, pad=12)
    fig.text(0.01, 0.01, "Gray: agent's own claim. Blue: verified by database state. Lines: 95% CI.",
             color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out_dir / "said_vs_did.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    # 2. How the two metrics pull apart as k grows.
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    xs = list(s["curve"].keys())
    for key, color, name in (("pass@k", ORANGE, "pass@k (at least once)"), ("pass^k", BLUE, "pass^k (every time)")):
        ys = [s["curve"][j][key] for j in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=name, zorder=3)
        ax.text(xs[-1] + 0.1, ys[-1], f"{name.split()[0]}  {ys[-1]:.0%}", color=INK, va="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.2, xs[-1] + 1.3)
    ax.set_xlabel("k (number of attempts)", color=INK_2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    ax.set_title("Trying more often looks better; succeeding every time looks worse",
                 loc="left", color=INK, fontsize=12, pad=12)
    fig.tight_layout()
    p = out_dir / "pass_vs_k.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="JSONL file from agent.run_bench")
    ap.add_argument("--out", default="results")
    ap.add_argument("--usd-per-mtok", type=float, default=0.0,
                    help="optional: price per million tokens, to estimate API cost")
    args = ap.parse_args()

    records = load(args.runs)
    s = summarize(records, args.usd_per_mtok)
    print(report(s))
    out = Path(args.out)
    write_csvs(s, out)
    figs = plot(s, records[0].get("model", "model"), out / "figures")
    print(f"\nWrote {out / 'main.csv'}, {out / 'per_task.csv'}, " + ", ".join(str(p) for p in figs))


if __name__ == "__main__":
    main()