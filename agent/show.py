"""Print one trajectory in readable form.

    python -m agent.show runs/baseline_k1.jsonl                       # list all attempts
    python -m agent.show runs/baseline_k1.jsonl refund_damaged_gold   # show one task
"""

import json
import sys


def main():
    path = sys.argv[1]
    records = [json.loads(line) for line in open(path)]
    if len(sys.argv) < 3:
        for r in records:
            v = "PASS" if r["verified_success"] else "FAIL"
            print(f"{r['task_id']:<32} attempt {r['attempt']}  {v}  claimed={r['claimed_success']}")
        return

    attempt = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    r = next(x for x in records if x["task_id"] == sys.argv[2] and x["attempt"] == attempt)
    print(f"TASK: {r['task_id']}  attempt {attempt}  ({r['finish_reason']}, {r['steps']} steps)\n")
    for m in r["messages"][1:]:  # skip the system prompt
        if m["role"] == "user":
            print(f"CUSTOMER: {m['content']}\n")
        elif m["role"] == "assistant":
            if m.get("content"):
                print(f"AGENT: {m['content'].strip()}")
            for c in m.get("tool_calls", []):
                print(f"  -> {c['function']['name']}({json.dumps(c['function']['arguments'])})")
        elif m["role"] == "tool":
            print(f"  <- {m['content'][:200]}")
    print("\nVERDICT:", "PASS" if r["verified_success"] else "FAIL",
          "| agent claimed success:", r["claimed_success"])
    for miss in r["checker"]["missing"]:
        print(f"  missing:   {miss['table']} {miss['action']} {miss.get('match') or miss.get('set') or ''}"
              f" (found {miss['found']})")
    for hit in r["checker"]["forbidden_hit"]:
        print(f"  forbidden: {hit['table']} {hit['action']} -> {[h['after'] for h in hit['hits']]}")
    for c in r["checker"]["diff"]:
        print(f"  db change: {c['table']} {c['action']} {c['after'] or c['before']}")


if __name__ == "__main__":
    main()