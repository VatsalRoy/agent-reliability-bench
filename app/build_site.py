"""Build the static results explorer for GitHub Pages.

    python -m app.build_site

Reads the run logs in runs/ and writes:
  docs/index.html   the page (copied from app/site/index.html)
  docs/data.js      all results the page shows, baked in (no server needed)
  docs/.nojekyll    tells GitHub Pages to serve the files as they are

Only runs/main_k5.jsonl is required. The control, probes, and judge files are
included when they exist.
"""

import argparse
import json
import shutil
from pathlib import Path

from eval.checker import effect_matches, load_tasks
from eval.failures import category, classify
from eval.judge import agreement
from eval.metrics import summarize
from eval.probes import compare

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "site" / "index.html"

RUNS = [  # (key, label, file name pattern with {k})
    ("main", "Main run", "main_k{k}.jsonl"),
    ("control", "Control: new seeds", "control_k{k}.jsonl"),
    ("rename", "Probe: renamed tools", "probe_rename_k{k}.jsonl"),
    ("reformat", "Probe: reformatted output", "probe_reformat_k{k}.jsonl"),
    ("misleading", "Probe: misleading names", "probe_misleading_k{k}.jsonl"),
]


# ---------- readable text for effects and changes ----------
def _fmt_val(v):
    if isinstance(v, dict):
        if "contains" in v:
            return f'contains "{v["contains"]}"'
        if "contains_any" in v:
            return "mentions " + " or ".join(f'"{x}"' for x in v["contains_any"])
        if "in" in v:
            return "one of " + ", ".join(map(str, v["in"]))
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmt_conds(c):
    return ", ".join(f"{k} {_fmt_val(v) if isinstance(v, dict) else '= ' + _fmt_val(v)}"
                     for k, v in (c or {}).items())


def describe_effect(e: dict) -> str:
    verb = {"insert": "new row in", "update": "change to", "delete": "deleted row in"}[e["action"]]
    text = f"{verb} {e['table']}"
    if e["action"] == "update":
        if e.get("where"):
            text += f" where {_fmt_conds(e['where'])}"
        if e.get("set"):
            text += f": set {_fmt_conds(e['set'])}"
    elif e.get("match"):
        text += f" with {_fmt_conds(e['match'])}"
    if e.get("count") is not None:
        text += f" (exactly {e['count']})"
    return text


def describe_change(c: dict) -> str:
    if c["action"] == "insert":
        row = {k: v for k, v in c["after"].items() if k != "id"}
        return f"{c['table']}: new row " + ", ".join(f"{k}={_fmt_val(v)}" for k, v in row.items())
    if c["action"] == "delete":
        return f"{c['table']} #{c['before']['id']}: deleted"
    parts = [f"{k} {_fmt_val(c['before'][k])} → {_fmt_val(c['after'][k])}" for k in c.get("changed", [])]
    return f"{c['table']} #{c['after']['id']}: " + ", ".join(parts)


# ---------- one attempt, slimmed for the page ----------
def _short(text: str, n: int = 700) -> str:
    return text if len(text) <= n else text[:n] + " …"


def events(record: dict) -> list[dict]:
    """The conversation as a list of events, with each tool call paired with its result."""
    out, calls = [], iter(record["tool_calls"])
    for m in record["messages"]:
        if m["role"] == "user":
            out.append({"t": "customer", "text": m["content"]})
        elif m["role"] == "assistant":
            if m.get("content", "").strip():
                out.append({"t": "agent", "text": m["content"].strip()})
            for c in m.get("tool_calls", []):
                logged = next(calls, None)
                result = logged["result"] if logged else {}
                out.append({"t": "call", "name": c["function"]["name"],
                            "args": c["function"]["arguments"],
                            "error": "error" in result,
                            "result": _short(json.dumps(result, default=str))})
    return out


def slim(record: dict, task: dict, judge: dict | None) -> dict:
    ck = record["checker"]
    changes = []
    for c in ck["diff"]:
        if any(effect_matches(e, c) for e in task.get("forbidden_effects", [])):
            kind = "forbidden"
        elif any(effect_matches(e, c) for e in task["required_effects"]):
            kind = "required"
        else:
            kind = "other"
        changes.append({"text": describe_change(c), "kind": kind})
    sig = classify(record)
    return {
        "attempt": record["attempt"],
        "pass": bool(record["verified_success"]),
        "claimed": bool(record["claimed_success"]),
        "outcome": record.get("outcome_tag"),
        "finish": record["finish_reason"],
        "steps": record["steps"],
        "seconds": record["seconds"],
        "tokens": record["prompt_tokens"] + record["completion_tokens"],
        "signature": sig,
        "category": category(record, sig) if sig else None,
        "missing": [describe_effect(m) + f" (found {m['found']})" for m in ck["missing"]],
        "changes": changes,
        "events": events(record),
        "judge": judge,
    }


# ---------- assemble ----------
def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def load_judges(runs_dir: Path) -> dict:
    judges = {}
    for view in ("full", "reply"):
        path = runs_dir / f"judge_{view}.jsonl"
        if path.exists():
            for j in load(path):
                judges.setdefault((j["task_id"], j["attempt"]), {})[view] = {
                    "verdict": j["judge"], "reason": j["reason"], "model": j.get("judge_model")}
    return judges


def pick_example(records: list[dict]) -> dict | None:
    """A false success claim that broke a rule, preferring the cancel-a-shipped-order task."""
    liars = [r for r in records if r["claimed_success"] and not r["verified_success"] and r["checker"]["forbidden_hit"]]
    liars.sort(key=lambda r: (r["task_id"] != "trap_cancel_shipped", r["task_id"], r["attempt"]))
    if not liars:
        return None
    r = liars[0]
    replies = [c["args"].get("message", "") for c in r["tool_calls"]
               if c.get("real_name") == "send_reply" and "error" not in c["result"]]
    return {"task_id": r["task_id"], "attempt": r["attempt"],
            "reply": replies[-1] if replies else r["final_text"],
            "outcome": r.get("outcome_tag"),
            "forbidden": [describe_change(h) for f in r["checker"]["forbidden_hit"] for h in f["hits"]]}


def build(runs_dir: Path, k: int) -> dict:
    tasks = load_tasks()
    by_id = {t["id"]: t for t in tasks}
    main_path = runs_dir / f"main_k{k}.jsonl"
    if not main_path.exists():
        raise SystemExit(f"{main_path} not found. Run the benchmark first (make run K={k}).")
    judges = load_judges(runs_dir)
    data = {"k": k, "tasks": [], "runs": [], "summary": {}, "judge": {}, "probes": []}

    for t in tasks:
        data["tasks"].append({
            "id": t["id"], "difficulty": t.get("difficulty"), "category": t.get("category"),
            "message": t["user_message"],
            "required": [describe_effect(e) for e in t["required_effects"]],
            "forbidden": [describe_effect(e) for e in t.get("forbidden_effects", [])],
        })

    main_records = None
    for key, label, pattern in RUNS:
        path = runs_dir / pattern.format(k=k)
        if not path.exists():
            continue
        records = load(path)
        if key == "main":
            main_records = records
        attempts = {}
        for r in records:
            j = judges.get((r["task_id"], r["attempt"])) if key == "main" else None
            attempts.setdefault(r["task_id"], []).append(slim(r, by_id[r["task_id"]], j))
        for a in attempts.values():
            a.sort(key=lambda x: x["attempt"])
        s = summarize(records)
        data["runs"].append({
            "key": key, "label": label, "model": records[0].get("model"),
            "variant": records[0].get("variant", "none"),
            "summary": {x: s[x] for x in ("pass@1", "pass@k", "pass^k", "claimed_rate",
                                          "verified_rate", "false_claims_among_claims")},
            "attempts": attempts,
        })
        if key != "main" and main_records is not None:
            c = compare(main_records, records, by_id)
            data["probes"].append({"key": key, "label": label,
                                   **{g: {"delta": c[g]["delta"], "ci": c[g]["ci"], "flips": c[g]["flips"],
                                          "effect": c[g]["effect"]} for g in ("state", "name")}})

    s = summarize(main_records)
    data["summary"] = {key: s[key] for key in ("pass@1", "pass@1_ci", "pass@k", "pass@k_ci", "pass^k",
                                               "pass^k_ci", "claimed_rate", "verified_rate",
                                               "false_claims_among_claims", "dependable_tasks", "tasks",
                                               "attempts")}
    data["summary"]["model"] = main_records[0].get("model")
    for view in ("full", "reply"):
        pairs = [(j[view]["verdict"] == "PASS", r["verified_success"])
                 for r in main_records
                 if (j := judges.get((r["task_id"], r["attempt"]))) and view in j]
        if pairs:
            a = agreement(pairs)
            model = next(j[view]["model"] for j in judges.values() if view in j)
            data["judge"][view] = {"model": model, **{x: a[x] for x in (
                "n", "kappa", "judge_pass_rate", "judge_passed_db_wrong", "wrong_among_judge_passes", "accuracy")}}
    data["example"] = pick_example(main_records)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()

    data = build(Path(args.runs), args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, out / "index.html")
    (out / "data.js").write_text("window.BENCH = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (out / ".nojekyll").write_text("")
    size = (out / "data.js").stat().st_size / 1e6
    runs = ", ".join(r["key"] for r in data["runs"])
    print(f"Wrote {out / 'index.html'} and {out / 'data.js'} ({size:.1f} MB; runs: {runs}; "
          f"judge views: {', '.join(data['judge']) or 'none'})")


if __name__ == "__main__":
    main()
