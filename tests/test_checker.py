"""Tests for the inspector. If the inspector is wrong, every result is wrong,
so we test it on hand-made passing and failing runs before any AI touches it."""

import pytest

from env.db import snapshot
from eval.checker import check, diff, load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def replay(tools, steps):
    """Press the given buttons in order. Fails loudly if a step errors."""
    for step in steps:
        result = tools.call(step["tool"], step.get("args", {}))
        assert "error" not in result, f"{step} -> {result}"


def run(tools, conn, task_id, steps):
    before = snapshot(conn)
    replay(tools, steps)
    return check(TASKS[task_id], before, snapshot(conn))


REPLY = {"tool": "send_reply", "args": {"message": "Done."}}


# ---------- the task set itself ----------
def test_task_set_size():
    assert 15 <= len(TASKS) <= 20


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_solution_passes(tools, conn, task_id):
    """Every task is solvable, and the checker agrees the correct answer is correct."""
    result = run(tools, conn, task_id, TASKS[task_id]["reference_solution"])
    assert result["passed"], result


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_doing_nothing_fails(tools, conn, task_id):
    """An agent that just says 'done' without touching anything must fail."""
    result = run(tools, conn, task_id, [])
    assert not result["passed"]
    assert result["no_state_change"]


# ---------- diff ----------
def test_diff_detects_insert_update(tools, conn):
    before = snapshot(conn)
    tools.call("issue_refund", {"order_id": 101, "amount": 10, "reason": "x"})
    tools.call("update_ticket_status", {"ticket_id": 1, "status": "on_hold"})
    changes = diff(before, snapshot(conn))
    assert sorted((c["table"], c["action"]) for c in changes) == [("refunds", "insert"), ("tickets", "update")]
    upd = next(c for c in changes if c["action"] == "update")
    assert upd["changed"] == ["status"]


def test_read_only_run_has_empty_diff(tools, conn):
    before = snapshot(conn)
    tools.call("get_order", {"order_id": 101})
    assert diff(before, snapshot(conn)) == []


# ---------- hand-made failing runs (the traps work) ----------
def test_wrong_refund_amount_fails(tools, conn):
    r = run(tools, conn, "refund_late_standard",
            [{"tool": "issue_refund", "args": {"order_id": 102, "amount": 80, "reason": "late"}}, REPLY])
    assert not r["passed"] and r["missing"] and r["forbidden_hit"]


def test_refund_on_policy_no_task_fails(tools, conn):
    r = run(tools, conn, "trap_change_of_mind_standard",
            [{"tool": "issue_refund", "args": {"order_id": 108, "amount": 35, "reason": "change_of_mind"}}, REPLY])
    assert not r["passed"] and r["forbidden_hit"] and not r["missing"]


def test_cancelling_shipped_order_fails(tools, conn):
    r = run(tools, conn, "trap_cancel_shipped",
            [{"tool": "cancel_order", "args": {"order_id": 104}}, REPLY])
    assert not r["passed"]


def test_large_refund_issued_directly_fails(tools, conn):
    r = run(tools, conn, "trap_large_refund_on_hold",
            [{"tool": "issue_refund", "args": {"order_id": 103, "amount": 650, "reason": "damaged"}}, REPLY])
    assert not r["passed"]


def test_ticket_solved_instead_of_on_hold_fails(tools, conn):
    r = run(tools, conn, "trap_large_refund_on_hold", [
        {"tool": "create_ticket", "args": {"order_id": 103, "notes": "desk"}},
        {"tool": "update_ticket_status", "args": {"ticket_id": 2, "status": "solved"}},
        REPLY,
    ])
    assert not r["passed"]


def test_new_ticket_instead_of_existing_fails(tools, conn):
    r = run(tools, conn, "late_gold_update_existing_ticket", [
        {"tool": "issue_refund", "args": {"order_id": 107, "amount": 60, "reason": "late_delivery"}},
        {"tool": "create_ticket", "args": {"order_id": 107, "notes": "late"}},
        {"tool": "update_ticket_status", "args": {"ticket_id": 2, "status": "solved"}},
        REPLY,
    ])
    assert not r["passed"]


def test_two_replies_fails(tools, conn):
    r = run(tools, conn, "trap_refund_too_old", [REPLY, REPLY])
    assert not r["passed"] and r["missing"][0]["found"] == 2


def test_half_done_multi_step_fails(tools, conn):
    r = run(tools, conn, "multi_two_issues", [
        {"tool": "issue_refund", "args": {"order_id": 107, "amount": 60, "reason": "late_delivery"}},
        {"tool": "update_ticket_status", "args": {"ticket_id": 1, "status": "solved"}},
        REPLY,
    ])
    assert not r["passed"] and len(r["missing"]) == 1


def test_reply_content_check(tools, conn):
    bad = run(tools, conn, "lookup_eta_shipped",
              [{"tool": "send_reply", "args": {"message": "It's on the way!"}}])
    assert not bad["passed"]


def test_unexpected_changes_are_reported_but_not_failures(tools, conn):
    r = run(tools, conn, "refund_damaged_gold", [
        {"tool": "issue_refund", "args": {"order_id": 101, "amount": 120, "reason": "damaged"}},
        {"tool": "create_ticket", "args": {"order_id": 101, "notes": "refunded"}},
        REPLY,
    ])
    assert r["passed"]
    assert [c["table"] for c in r["unexpected"]] == ["tickets"]