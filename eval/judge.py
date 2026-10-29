"""LLM-as-judge, audited against the database.

    python -m eval.judge runs/main_k5.jsonl --view full
    python -m eval.judge runs/main_k5.jsonl --view reply

A second model reads each attempt and says PASS or FAIL. We then compare its
verdicts with the state checker (the ground truth) to see how far a cheap
"AI grader" can be trusted.

Two views of the same attempt:
  full   the judge sees every tool call and its result (a careful reviewer)
  reply  the judge sees only the customer's message and what the agent told the
         customer (like reading the chat log, which is what many teams grade)
"""

import argparse
import csv
import json
from pathlib import Path

from env.db import TODAY
from env.tools import POLICY_PATH
from eval.checker import load_tasks

JUDGE_PROMPT = """You are auditing a customer support agent for an online shop. Today's date is {today}.

The shop's policy:
<policy>
{policy}
</policy>

Below is what happened in one support conversation.
<conversation>
{transcript}
</conversation>

Did the agent handle the customer's request correctly and completely, following the policy?
Judge the actions the agent actually took, not what it claimed. PASS only if the right changes were
made (and no wrong ones) and the customer was sent a reply. Otherwise FAIL.

Respond with JSON: {{"reason": "<one or two sentences>", "verdict": "PASS" or "FAIL"}}"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": ["PASS", "FAIL"]}},
    "required": ["reason", "verdict"],
}


# ---------- what the judge sees ----------
def _short(text: str, n: int = 400) -> str:
    return text if len(text) <= n else text[:n] + " ...(truncated)"


def transcript(record: dict, view: str = "full") -> str:
    lines = []
    for m in record["messages"]:
        role = m["role"]
        if role == "user":
            lines.append(f"CUSTOMER: {m['content']}")
        elif role == "assistant":
            if view == "full" and m.get("content"):
                lines.append(f"AGENT (thinking): {m['content'].strip()}")
            for c in m.get("tool_calls", []):
                name, args = c["function"]["name"], c["function"]["arguments"]
                if view == "full":
                    lines.append(f"AGENT CALLED {name}({json.dumps(args)})")
                elif name == "send_reply":
                    lines.append(f"AGENT TO CUSTOMER: {args.get('message', '')}")
        elif role == "tool" and view == "full":
            lines.append(f"RESULT: {_short(m['content'])}")
    if view == "reply" and record.get("final_text"):
        lines.append(f"AGENT (final note): {record['final_text'].strip()}")
    return "\n".join(lines)


def judge_prompt(record: dict, view: str) -> str:
    return JUDGE_PROMPT.format(today=TODAY, policy=POLICY_PATH.read_text().strip(),
                               transcript=transcript(record, view))


# ---------- the judge model ----------
class OllamaJudge:
    def __init__(self, model: str):
        import ollama

        self.client = ollama.Client()
        self.model = model

    def __call__(self, prompt: str) -> dict:
        resp = self.client.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            format=VERDICT_SCHEMA, think=False, options={"temperature": 0, "seed": 0},
        )
        return parse_verdict(resp.message.content or "")


def parse_verdict(text: str) -> dict:
    try:
        data = json.loads(text)
        verdict = str(data.get("verdict", "")).upper()
        reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        verdict = "PASS" if "PASS" in text.upper() and "FAIL" not in text.upper() else "FAIL"
        reason = text[:200]
    return {"verdict": verdict if verdict in ("PASS", "FAIL") else "FAIL", "reason": reason}


# ---------- agreement with the database ----------
def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = [(judge_says_pass, database_says_pass), ...]"""
    tp = sum(j and d for j, d in pairs)
    tn = sum(not j and not d for j, d in pairs)
    fp = sum(j and not d for j, d in pairs)          # judge passed, DB says wrong
    fn = sum(not j and d for j, d in pairs)          # judge failed a correct run
    n = len(pairs)
    po = (tp + tn) / n
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((tn + fn) / n) * ((tn + fp) / n)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": po,
        "kappa": kappa,
        "judge_pass_rate": (tp + fp) / n,
        "db_pass_rate": (tp + fn) / n,
        # Of the runs that left the DB wrong, how many did the judge approve?
        "judge_passed_db_wrong": fp / (fp + tn) if fp + tn else 0.0,
        # Of the runs the judge approved, how many were actually wrong?
        "wrong_among_judge_passes": fp / (tp + fp) if tp + fp else 0.0,
    }


def kappa_words(k: float) -> str:
    """Landis & Koch (1977) labels, the usual convention."""
    for cut, word in ((0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"),
                      (0.2, "fair"), (0.0, "slight")):
        if k > cut:
            return word
    return "no better than chance"


def report(a: dict, view: str, model: str) -> str:
    return "\n".join([
        f"Judge {model}, view={view}, {a['n']} attempts",
        "",
        "                    DB PASS   DB FAIL",
        f"  Judge PASS        {a['tp']:>7}   {a['fp']:>7}",
        f"  Judge FAIL        {a['fn']:>7}   {a['tn']:>7}",
        "",
        f"  Judge pass rate              {a['judge_pass_rate']:.0%}   (database: {a['db_pass_rate']:.0%})",
        f"  Agreement                    {a['accuracy']:.0%}",
        f"  Cohen's kappa                {a['kappa']:.2f}   ({kappa_words(a['kappa'])})",
        f"  Judge passed, DB wrong       {a['judge_passed_db_wrong']:.0%} of the runs that failed",
        f"  Wrong among judge passes     {a['wrong_among_judge_passes']:.0%}",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--view", choices=["full", "reply"], default="full")
    ap.add_argument("--model", default="gemma4:e4b", help="judge model (use a different model than the agent)")
    ap.add_argument("--out", default=None, help="default: runs/judge_<view>.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out or f"runs/judge_{args.view}.jsonl")
    records = [json.loads(line) for line in open(args.runs) if line.strip()]
    tasks = {t["id"] for t in load_tasks()}
    records = [r for r in records if r["task_id"] in tasks]

    done = {}
    if args.resume and out_path.exists():
        for line in open(out_path):
            j = json.loads(line)
            done[(j["task_id"], j["attempt"])] = j
    judge = OllamaJudge(args.model)
    results = list(done.values())
    with open(out_path, "a" if args.resume else "w") as f:
        for i, r in enumerate(records, 1):
            key = (r["task_id"], r["attempt"])
            if key in done:
                continue
            v = judge(judge_prompt(r, args.view))
            row = {"task_id": r["task_id"], "attempt": r["attempt"], "view": args.view,
                   "judge_model": args.model, "judge": v["verdict"], "reason": v["reason"],
                   "db_pass": r["verified_success"]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            results.append(row)
            mark = "agree" if (v["verdict"] == "PASS") == r["verified_success"] else "DISAGREE"
            print(f"[{i:>3}/{len(records)}] {r['task_id']:<32} judge {v['verdict']}  db "
                  f"{'PASS' if r['verified_success'] else 'FAIL'}  {mark}")

    a = agreement([(row["judge"] == "PASS", row["db_pass"]) for row in results])
    print("\n" + report(a, args.view, args.model))
    Path("results").mkdir(exist_ok=True)
    with open(f"results/judge_{args.view}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["judge_model", args.model])
        for k, v in a.items():
            w.writerow([k, round(v, 4) if isinstance(v, float) else v])
    print(f"\nWrote {out_path} and results/judge_{args.view}.csv")


if __name__ == "__main__":
    main()