"""The inspector: grades a run by what actually changed in the database.

How it works:
  1. diff(before, after) lists every row that was inserted, updated, or deleted.
  2. check(task, before, after) tests that diff against the task's rules:
       - every required effect happened (exactly `count` times if given)
       - no forbidden effect happened
  The diff explains a failure; the checks decide it.

Effect spec (in the task YAML):
  table:  refunds | tickets | orders | replies | customers
  action: insert | update | delete
  match:  {field: value}   for insert -> the new row; for delete -> the removed row
  where:  {field: value}   for update -> the row before the change
  set:    {field: value}   for update -> the row after the change
  count:  N                (optional, required effects only) exactly N matches

A value can be a plain value (equal; numbers within 0.01) or an operator:
  {contains: "text"}           case-insensitive substring
  {contains_any: ["a", "b"]}   at least one substring present
  {in: [v1, v2]}               one of these values
"""

from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
ACTIONS = ("insert", "update", "delete")


# ---------- diff ----------
def diff(before: dict, after: dict) -> list[dict]:
    """Row-level diff between two snapshots, matched by primary key `id`."""
    changes = []
    for table in after:
        old = {r["id"]: r for r in before.get(table, [])}
        new = {r["id"]: r for r in after[table]}
        for rid in sorted(new.keys() - old.keys()):
            changes.append({"table": table, "action": "insert", "before": None, "after": new[rid]})
        for rid in sorted(old.keys() - new.keys()):
            changes.append({"table": table, "action": "delete", "before": old[rid], "after": None})
        for rid in sorted(old.keys() & new.keys()):
            if old[rid] != new[rid]:
                changed = sorted(k for k in new[rid] if old[rid].get(k) != new[rid][k])
                changes.append({"table": table, "action": "update",
                                "before": old[rid], "after": new[rid], "changed": changed})
    return changes


# ---------- matching ----------
def _value_ok(actual, expected) -> bool:
    if isinstance(expected, dict):
        text = str(actual).lower()
        if "contains" in expected:
            return str(expected["contains"]).lower() in text
        if "contains_any" in expected:
            return any(str(s).lower() in text for s in expected["contains_any"])
        if "in" in expected:
            return any(_value_ok(actual, v) for v in expected["in"])
        raise ValueError(f"Unknown operator in {expected}")
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= 0.01
        except (TypeError, ValueError):
            return False
    return actual == expected


def _row_ok(row: dict | None, conditions: dict | None) -> bool:
    if not conditions:
        return True
    if row is None:
        return False
    return all(k in row and _value_ok(row[k], v) for k, v in conditions.items())


def effect_matches(effect: dict, change: dict) -> bool:
    if change["table"] != effect["table"] or change["action"] != effect["action"]:
        return False
    if effect["action"] == "insert":
        return _row_ok(change["after"], effect.get("match"))
    if effect["action"] == "delete":
        return _row_ok(change["before"], effect.get("match"))
    return _row_ok(change["before"], effect.get("where")) and _row_ok(change["after"], effect.get("set"))


# ---------- check ----------
def check(task: dict, before: dict, after: dict) -> dict:
    changes = diff(before, after)
    missing, forbidden_hit = [], []

    for eff in task.get("required_effects", []):
        n = sum(effect_matches(eff, c) for c in changes)
        want = eff.get("count")
        if (want is None and n == 0) or (want is not None and n != want):
            missing.append({**eff, "found": n})

    for eff in task.get("forbidden_effects", []):
        hits = [c for c in changes if effect_matches(eff, c)]
        if hits:
            forbidden_hit.append({**eff, "hits": hits})

    explained = [c for c in changes
                 if any(effect_matches(e, c) for e in task.get("required_effects", []))]
    unexpected = [c for c in changes if c not in explained]

    return {
        "task_id": task["id"],
        "passed": not missing and not forbidden_hit,
        "missing": missing,            # required effects that didn't happen
        "forbidden_hit": forbidden_hit,  # forbidden effects that did happen
        "unexpected": unexpected,      # other changes (informational, not a failure)
        "no_state_change": not changes,
        "diff": changes,
    }


# ---------- loading tasks ----------
def validate_task(task: dict) -> None:
    for key in ("id", "user_message", "required_effects"):
        if key not in task:
            raise ValueError(f"Task {task.get('id', '?')} is missing '{key}'")
    for eff in task["required_effects"] + task.get("forbidden_effects", []):
        if eff.get("action") not in ACTIONS:
            raise ValueError(f"Task {task['id']}: bad action in {eff}")
        if "table" not in eff:
            raise ValueError(f"Task {task['id']}: effect has no table: {eff}")


def load_tasks(tasks_dir: Path = TASKS_DIR) -> list[dict]:
    """Load every task from tasks/*.yaml. A file may hold one task or a list."""
    tasks = []
    for path in sorted(Path(tasks_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        tasks.extend(data if isinstance(data, list) else [data])
    ids = [t["id"] for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate task ids: {dupes}")
    for t in tasks:
        validate_task(t)
    return tasks