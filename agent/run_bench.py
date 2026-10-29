"""Run the benchmark: every task, k times, each on a fresh copy of the shop.

    python -m agent.run_bench --k 1 --out runs/baseline_k1.jsonl

Writes one JSON line per attempt (a "trajectory"). Infrastructure errors
(e.g. Ollama crashed) go to runs/errors.jsonl and are left out of the scores,
because they say nothing about the agent.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.react_agent import OllamaModel, prompt_hash, run_attempt
from agent.variants import BASE, VARIANTS
from eval.checker import load_tasks


def run(tasks, make_model, *, k: int, out_path: Path, meta: dict, max_steps: int = 15,
        seed_base: int = 0, quiet: bool = False, resume: bool = False, variant=BASE) -> list[dict]:
    """make_model(seed) returns a model_fn. Returns the list of graded records.

    resume=True keeps attempts already saved in out_path and runs only the
    missing ones, so an interrupted long run can pick up where it stopped.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.parent / "errors.jsonl"
    records = []
    if resume and out_path.exists():
        records = [json.loads(line) for line in open(out_path) if line.strip()]
        other = {r.get("prompt_hash") for r in records} - {meta.get("prompt_hash")}
        if other:
            raise SystemExit(f"Can't resume: {out_path} was made with a different setup {other}. "
                             "Use a new --out file.")
    done = {(r["task_id"], r["attempt"]) for r in records}
    total = len(tasks) * k
    with open(out_path, "a" if resume else "w") as out, open(err_path, "a") as err:
        n = 0
        for task in tasks:
            for attempt in range(k):
                n += 1
                if (task["id"], attempt) in done:
                    continue
                seed = seed_base + attempt
                rec = run_attempt(task, make_model(seed), max_steps=max_steps, variant=variant)
                rec.update(meta, attempt=attempt, seed=seed)
                line = json.dumps(rec, default=str)
                if rec["finish_reason"] == "infra_error":
                    err.write(line + "\n")
                    err.flush()
                else:
                    out.write(line + "\n")
                    out.flush()  # keep results even if the run is interrupted
                    records.append(rec)
                if not quiet:
                    print(_progress(n, total, rec))
    return records


def _progress(n, total, rec) -> str:
    if rec["finish_reason"] == "infra_error":
        return f"[{n:>3}/{total}] {rec['task_id']:<32} INFRA ERROR  {rec['error']}"
    verdict = "PASS" if rec["verified_success"] else "FAIL"
    said = "said done" if rec["claimed_success"] else "said failed"
    flag = "  <-- said done, DB disagrees" if rec["claimed_success"] and not rec["verified_success"] else ""
    return (f"[{n:>3}/{total}] {rec['task_id']:<32} {verdict}  ({said}, {rec['steps']} steps, "
            f"{rec['seconds']:.1f}s){flag}")


def summarize(records: list[dict]) -> str:
    if not records:
        return "No completed attempts."
    n = len(records)
    verified = sum(r["verified_success"] for r in records)
    claimed = sum(r["claimed_success"] for r in records)
    lied = sum(r["claimed_success"] and not r["verified_success"] for r in records)
    max_steps = sum(r["finish_reason"] == "max_steps" for r in records)
    secs = sum(r["seconds"] for r in records)
    toks = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    return "\n".join([
        "",
        f"Attempts:                    {n}",
        f"Agent SAID it succeeded:     {claimed}/{n} ({claimed / n:.0%})",
        f"Database CONFIRMS success:   {verified}/{n} ({verified / n:.0%})",
        f"Said done but DB disagrees:  {lied}/{n} ({lied / n:.0%})",
        f"Hit the step limit:          {max_steps}",
        f"Total time / tokens:         {secs / 60:.1f} min / {toks:,} tokens",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.5:4b-mlx")
    p.add_argument("--k", type=int, default=1, help="attempts per task")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed-base", type=int, default=0, help="attempt i uses seed seed_base + i")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--out", default="runs/baseline_k1.jsonl")
    p.add_argument("--resume", action="store_true", help="skip attempts already saved in --out")
    p.add_argument("--variant", default="none", choices=sorted(VARIANTS),
                   help="probe: change what the agent is shown (see agent/variants.py)")
    args = p.parse_args()

    tasks = load_tasks()
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            raise SystemExit(f"Unknown task ids: {sorted(missing)}")

    variant = VARIANTS[args.variant]
    meta = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model": args.model,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "variant": args.variant,
        "seed_base": args.seed_base,
        "prompt_hash": prompt_hash(variant.schemas(), variant.fingerprint()),
        "platform": platform.platform(),
    }
    print(f"Model {args.model} | {len(tasks)} tasks x k={args.k} | variant {args.variant} | "
          f"seeds {args.seed_base}-{args.seed_base + args.k - 1} | setup {meta['prompt_hash']}\n")

    t0 = time.time()
    records = run(
        tasks,
        lambda seed: OllamaModel(args.model, temperature=args.temperature, seed=seed),
        k=args.k, out_path=Path(args.out), meta=meta,
        max_steps=args.max_steps, seed_base=args.seed_base, resume=args.resume, variant=variant,
    )
    print(summarize(records))
    print(f"\nSaved {len(records)} trajectories to {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()