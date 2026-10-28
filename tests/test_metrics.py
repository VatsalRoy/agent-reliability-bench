"""Tests for the reliability math, checked against hand-computed values."""

import json
from math import isclose

import pytest

from agent.run_bench import run
from eval.checker import load_tasks
from eval.metrics import (bootstrap_mean, pass_at_k, pass_hat_k, per_task, plot, report,
                          summarize, wilson, write_csvs)
from tests.test_agent import perfect_model

TASKS = {t["id"]: t for t in load_tasks()}


# ---------- estimators ----------
def test_pass_at_k_by_hand():
    # 5 attempts, 2 successes, k=2: P(both picks fail) = C(3,2)/C(5,2) = 3/10
    assert isclose(pass_at_k(5, 2, 2), 1 - 3 / 10)
    assert pass_at_k(5, 0, 5) == 0.0
    assert pass_at_k(5, 1, 5) == 1.0          # one success in five is "at least once"
    assert pass_at_k(5, 4, 1) == 4 / 5        # pass@1 is just the success rate


def test_pass_hat_k_by_hand():
    # all 3 picks succeed: C(4,3)/C(5,3) = 4/10
    assert isclose(pass_hat_k(5, 4, 3), 4 / 10)
    assert pass_hat_k(5, 5, 5) == 1.0
    assert pass_hat_k(5, 4, 5) == 0.0         # one failure ruins "every time"
    assert pass_hat_k(5, 3, 1) == 3 / 5


def test_ordering_always_holds():
    for n in range(1, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                assert pass_hat_k(n, c, k) <= c / n <= pass_at_k(n, c, k) + 1e-12


def test_wilson():
    lo, hi = wilson(10, 20)
    assert lo < 0.5 < hi and isclose(lo + hi, 1.0)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_bootstrap_is_reproducible_and_brackets_mean():
    vals = [0, 0.2, 0.4, 1, 1, 0.8]
    assert bootstrap_mean(vals) == bootstrap_mean(vals)
    lo, hi = bootstrap_mean(vals)
    assert lo <= sum(vals) / len(vals) <= hi


# ---------- summaries on fake records ----------
def rec(task, ok, claimed=True, tag="SUCCESS", secs=10.0, toks=1000, difficulty="easy"):
    return {"task_id": task, "difficulty": difficulty, "category": "x", "verified_success": ok,
            "claimed_success": claimed, "outcome_tag": tag, "seconds": secs,
            "prompt_tokens": toks, "completion_tokens": 0, "model": "fake"}


@pytest.fixture
def records():
    # task A: 5/5, task B: 3/5, task C: 0/5 (claims success anyway, no tag)
    return ([rec("A", True)] * 5
            + [rec("B", True)] * 3 + [rec("B", False)] * 2
            + [rec("C", False, tag=None, difficulty="hard")] * 5)


def test_summary_numbers(records):
    s = summarize(records)
    assert s["k"] == 5 and s["tasks"] == 3 and s["attempts"] == 15
    assert isclose(s["pass@1"], (1 + 0.6 + 0) / 3)
    assert isclose(s["pass@k"], 2 / 3)        # A and B succeeded at least once
    assert isclose(s["pass^k"], 1 / 3)        # only A succeeded every time
    assert s["dependable_tasks"] == 1
    assert isclose(s["claimed_rate"], 1.0)
    assert isclose(s["claimed_explicit_rate"], 10 / 15)
    assert isclose(s["claimed_inferred_rate"], 5 / 15)
    assert isclose(s["false_claim_rate"], 7 / 15)
    assert isclose(s["seconds_per_dependable_task"], 150.0)
    assert s["by_difficulty"]["hard"]["pass@1"] == 0.0


def test_curve_starts_together_and_pulls_apart(records):
    c = summarize(records)["curve"]
    assert isclose(c[1]["pass@k"], c[1]["pass^k"])
    assert c[5]["pass@k"] > c[1]["pass@k"] and c[5]["pass^k"] < c[1]["pass^k"]


def test_uneven_attempts_rejected(records):
    with pytest.raises(ValueError):
        summarize(records[:-1])


def test_outputs_are_written(records, tmp_path):
    s = summarize(records, usd_per_mtok=0.5)
    assert "pass^5" in report(s) and "$" in report(s)
    write_csvs(s, tmp_path)
    assert (tmp_path / "main.csv").read_text().startswith("metric,value")
    assert len((tmp_path / "per_task.csv").read_text().splitlines()) == 4
    for p in plot(s, "fake", tmp_path / "figures"):
        assert p.stat().st_size > 10_000


def test_per_task_counts(records):
    rows = {r["task_id"]: r for r in per_task(records)}
    assert (rows["B"]["n"], rows["B"]["c"]) == (5, 3)


# ---------- resuming an interrupted run ----------
def test_resume_skips_done_attempts(tmp_path):
    task = TASKS["refund_damaged_gold"]
    out = tmp_path / "r.jsonl"
    meta = {"model": "fake", "prompt_hash": "abc"}
    run([task], lambda seed: perfect_model(task), k=2, out_path=out, meta=meta, quiet=True)
    lines = out.read_text().splitlines()
    out.write_text(lines[0] + "\n")                     # pretend it crashed after attempt 0

    calls = []

    def make(seed):
        calls.append(seed)
        return perfect_model(task)

    recs = run([task], make, k=2, out_path=out, meta=meta, quiet=True, resume=True)
    assert calls == [1]                                 # only the missing attempt ran
    assert sorted(r["attempt"] for r in recs) == [0, 1]
    assert len(out.read_text().splitlines()) == 2


def test_resume_refuses_mixed_setups(tmp_path):
    task = TASKS["refund_damaged_gold"]
    out = tmp_path / "r.jsonl"
    run([task], lambda s: perfect_model(task), k=1, out_path=out,
        meta={"prompt_hash": "old"}, quiet=True)
    with pytest.raises(SystemExit):
        run([task], lambda s: perfect_model(task), k=1, out_path=out,
            meta={"prompt_hash": "new"}, quiet=True, resume=True)


def test_records_from_runner_feed_metrics(tmp_path):
    task = TASKS["trap_cancel_shipped"]
    out = tmp_path / "r.jsonl"
    run([task], lambda s: perfect_model(task), k=3, out_path=out, meta={"model": "fake"}, quiet=True)
    s = summarize([json.loads(x) for x in out.read_text().splitlines()])
    assert s["pass^k"] == 1.0