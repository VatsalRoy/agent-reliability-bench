"""Tests for the static site builder and the dataset packager."""

import json

import pytest
from huggingface_hub import DatasetCard

from agent.run_bench import run
from app import build_dataset
from app.build_site import build, describe_change, describe_effect
from eval.checker import load_tasks
from tests.test_agent import Scripted, out, perfect_model

TASKS = load_tasks()
BY_ID = {t["id"]: t for t in TASKS}


def liar(task):
    return Scripted([out(calls=[("cancel_order", {"order_id": 104})]),
                     out(calls=[("send_reply", {"message": "Cancelled!"})]),
                     out("OUTCOME: SUCCESS")])


@pytest.fixture
def runs_dir(tmp_path):
    order = [(t["id"], a) for t in TASKS for a in range(2)]
    it = iter(order)

    def make(seed):
        tid, a = next(it)
        return liar(BY_ID[tid]) if tid == "trap_cancel_shipped" and a == 0 else perfect_model(BY_ID[tid])

    run(TASKS, make, k=2, out_path=tmp_path / "main_k2.jsonl", meta={"model": "fake"}, quiet=True)
    recs = [json.loads(x) for x in (tmp_path / "main_k2.jsonl").read_text().splitlines()]
    with open(tmp_path / "judge_full.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"task_id": r["task_id"], "attempt": r["attempt"], "judge": "PASS",
                                "reason": "ok", "judge_model": "fake-judge", "db_pass": r["verified_success"]}) + "\n")
    return tmp_path


def test_describe_effect_and_change():
    assert describe_effect({"table": "refunds", "action": "insert", "count": 1,
                            "match": {"order_id": 101, "amount": 120.0}}) == \
        "new row in refunds with order_id = 101, amount = 120.00 (exactly 1)"
    assert describe_effect({"table": "orders", "action": "update", "where": {"id": 105},
                            "set": {"status": "cancelled"}}) == \
        "change to orders where id = 105: set status = cancelled"
    change = {"table": "orders", "action": "update", "changed": ["status"],
              "before": {"id": 104, "status": "shipped"}, "after": {"id": 104, "status": "cancelled"}}
    assert describe_change(change) == "orders #104: status shipped → cancelled"


def test_build_site_data(runs_dir):
    d = build(runs_dir, k=2)
    assert [r["key"] for r in d["runs"]] == ["main"]
    assert len(d["tasks"]) == len(TASKS)
    assert d["example"]["task_id"] == "trap_cancel_shipped"
    assert d["example"]["forbidden"] == ["orders #104: status shipped → cancelled"]
    assert d["example"]["reply"] == "Cancelled!"
    att = d["runs"][0]["attempts"]["trap_cancel_shipped"][0]
    assert not att["pass"] and att["claimed"] and att["signature"] == "policy_violation"
    assert att["changes"][0]["kind"] == "forbidden"
    assert att["judge"]["full"]["verdict"] == "PASS"
    assert [e["t"] for e in att["events"]] == ["customer", "call", "call", "agent"]
    assert d["judge"]["full"]["judge_passed_db_wrong"] == 1.0
    json.dumps(d)


def test_build_site_needs_main_run(tmp_path):
    with pytest.raises(SystemExit):
        build(tmp_path, k=5)


def test_dataset_package(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_dataset", "--out", str(tmp_path / "ds")])
    build_dataset.main()
    rows = [json.loads(x) for x in (tmp_path / "ds" / "data" / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == len(TASKS)
    assert all(isinstance(r["required_effects"], str) for r in rows)
    assert json.loads(rows[0]["reference_solution"])
    card = DatasetCard.load(tmp_path / "ds" / "README.md")
    assert card.data.license == "mit"
    assert card.data.configs[0]["data_files"] == "data/tasks.jsonl"
    assert (tmp_path / "ds" / "raw" / "policy.md").exists()
