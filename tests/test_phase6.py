"""Tests for Phase 6: probe variants, the LLM judge plumbing, failure
signatures, and the probe comparison. No real model needed."""

import json

import pytest

from agent.react_agent import prompt_hash, run_attempt
from agent.variants import BASE, VARIANTS, Variant
from eval.checker import load_tasks
from eval.failures import category, classify, report as fail_report, tally, write_outputs as fail_write
from eval.judge import agreement, judge_prompt, parse_verdict, transcript
from eval.probes import compare, name_match_grade, report as probe_report, write_outputs as probe_write
from tests.test_agent import Scripted, out

TASKS = {t["id"]: t for t in load_tasks()}
REPLY = ("send_reply", {"message": "Done."})


def perfect_for(task, variant):
    """A perfect agent that uses whatever names the variant shows."""
    steps = [out(calls=[(variant.shown_name(s["tool"]), s["args"])]) for s in task["reference_solution"]]
    return Scripted(steps + [out("OUTCOME: SUCCESS")])


def attempt(task_id, calls, variant=BASE, final="OUTCOME: SUCCESS", **kw):
    steps = [out(calls=[c]) for c in calls] + [out(final)]
    rec = run_attempt(TASKS[task_id], Scripted(steps), variant=variant, **kw)
    rec["attempt"] = 0
    return rec


# ---------- variants ----------
def test_base_variant_keeps_old_fingerprint():
    assert BASE.fingerprint() == ""
    assert prompt_hash(BASE.schemas(), BASE.fingerprint()) == prompt_hash()


def test_every_variant_has_its_own_fingerprint():
    hashes = {prompt_hash(v.schemas(), v.fingerprint()) for v in VARIANTS.values()}
    assert len(hashes) == len(VARIANTS)


def test_rename_variant_names():
    v = VARIANTS["rename"]
    names = [s["function"]["name"] for s in v.schemas()]
    assert "lookup_order_record" in names and "get_order" not in names
    assert v.real_name("lookup_order_record") == "get_order"
    assert v.real_name("get_order") is None


def test_misleading_swap_is_one_pass():
    v = VARIANTS["misleading"]
    assert v.rename_text("call get_order then issue_refund") == "call issue_refund then get_order"
    assert v.real_name("issue_refund") == "get_order"


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        Variant("bad", {"get_order": "send_reply"})


def test_reformat_is_pretty_and_sorted():
    text = VARIANTS["reformat"].format_result({"b": 1, "a": 2})
    assert text.index('"a"') < text.index('"b"') and "\n" in text


def test_error_messages_use_shown_names():
    text = VARIANTS["rename"].format_result({"error": "Bad arguments for issue_refund: x"})
    assert "process_refund_payment" in text and "issue_refund" not in text


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("task_id", ["refund_damaged_gold", "trap_large_refund_on_hold", "multi_two_issues"])
def test_perfect_agent_passes_under_every_variant(variant, task_id):
    """The probes change only the presentation, so a perfect agent still passes."""
    v = VARIANTS[variant]
    rec = run_attempt(TASKS[task_id], perfect_for(TASKS[task_id], v), variant=v)
    assert rec["verified_success"] and rec["tool_errors"] == 0


def test_agent_sees_renamed_prompt_and_tools():
    v = VARIANTS["rename"]
    model = perfect_for(TASKS["refund_damaged_gold"], v)
    run_attempt(TASKS["refund_damaged_gold"], model, variant=v)
    system = model.seen[0][0]["content"]
    assert "lookup_order_record" in system and "get_order" not in system


def test_real_name_is_unknown_under_rename():
    rec = attempt("refund_damaged_gold", [("get_order", {"order_id": 101})], variant=VARIANTS["rename"])
    assert rec["tool_errors"] == 1 and "Unknown tool" in rec["tool_calls"][0]["result"]["error"]


# ---------- judge plumbing ----------
@pytest.fixture
def liar():
    return attempt("refund_damaged_gold", [("get_order", {"order_id": 101}),
                                           ("send_reply", {"message": "Your $120 refund is issued!"})])


def test_full_view_shows_actions(liar):
    t = transcript(liar, "full")
    assert "AGENT CALLED get_order" in t and "RESULT:" in t


def test_reply_view_shows_only_what_customer_saw(liar):
    t = transcript(liar, "reply")
    assert "Your $120 refund is issued!" in t
    assert "RESULT:" not in t and "get_order" not in t


def test_judge_prompt_has_policy_and_date(liar):
    p = judge_prompt(liar, "full")
    assert "Large refunds" in p and "2026-10-20" in p


def test_parse_verdict():
    assert parse_verdict('{"reason": "ok", "verdict": "PASS"}')["verdict"] == "PASS"
    assert parse_verdict("garbage")["verdict"] == "FAIL"
    assert parse_verdict('{"verdict": "maybe"}')["verdict"] == "FAIL"


def test_agreement_by_hand():
    perfect = agreement([(True, True), (False, False)] * 5)
    assert perfect["kappa"] == 1.0 and perfect["accuracy"] == 1.0
    rubber_stamp = agreement([(True, True)] * 6 + [(True, False)] * 4)
    assert rubber_stamp["kappa"] == 0.0                     # always says PASS: no skill
    assert rubber_stamp["judge_passed_db_wrong"] == 1.0
    assert rubber_stamp["wrong_among_judge_passes"] == 0.4
    # 2x2: tp=4 fp=1 fn=1 tn=4 -> po=0.8, pe=0.5, kappa=0.6
    a = agreement([(True, True)] * 4 + [(True, False)] + [(False, True)] + [(False, False)] * 4)
    assert abs(a["kappa"] - 0.6) < 1e-9


# ---------- failure signatures ----------
def test_signatures():
    cases = {
        "policy_violation": attempt("trap_change_of_mind_standard",
                                    [("issue_refund", {"order_id": 108, "amount": 35, "reason": "x"}), REPLY]),
        "no_reply_sent": attempt("refund_damaged_gold",
                                 [("issue_refund", {"order_id": 101, "amount": 120, "reason": "damaged"})]),
        "wrong_value": attempt("refund_damaged_gold",
                               [("issue_refund", {"order_id": 101, "amount": 100, "reason": "damaged"}), REPLY]),
        "duplicate_effect": attempt("trap_refund_too_old", [REPLY, REPLY]),
        "missing_effect": attempt("late_gold_update_existing_ticket",
                                  [("issue_refund", {"order_id": 107, "amount": 60, "reason": "late"}), REPLY]),
        "no_state_change": attempt("refund_damaged_gold", []),
        "tool_error_unrecovered": attempt("refund_damaged_gold",
                                          [("issue_refund", {"order_id": 101}), REPLY]),
    }
    for expected, rec in cases.items():
        assert classify(rec) == expected, (expected, classify(rec))


def test_step_limit_signature():
    model = Scripted([out(calls=[("get_order", {"order_id": 101})])] * 3)
    rec = run_attempt(TASKS["refund_damaged_gold"], model, max_steps=3)
    assert classify(rec) == "step_limit"


def test_passing_run_has_no_signature():
    rec = attempt("refund_damaged_gold", [("issue_refund", {"order_id": 101, "amount": 120, "reason": "d"}), REPLY])
    assert classify(rec) is None


def test_thinkingbox_categories():
    def cat(task_id, calls):
        rec = attempt(task_id, calls)
        return category(rec, classify(rec))

    refund = ("issue_refund", {"order_id": 101, "amount": 120, "reason": "damaged"})
    assert cat("trap_change_of_mind_standard",
               [("issue_refund", {"order_id": 108, "amount": 35, "reason": "x"}), REPLY]) == "wrong_state_update"
    assert cat("refund_damaged_gold", [refund]) == "incomplete_resolution"               # no reply
    assert cat("lookup_status_processing", []) == "incomplete_resolution"                # answered in chat
    assert cat("refund_damaged_gold", []) == "no_state_changing_action"                 # refund never made
    assert cat("late_gold_update_existing_ticket",
               [("issue_refund", {"order_id": 107, "amount": 60, "reason": "late"}), REPLY]) == "no_state_changing_action"
    assert cat("refund_damaged_gold", [("issue_refund", {"order_id": 101}), REPLY]) == "tool_usage"
    assert cat("lookup_eta_shipped", [("send_reply", {"message": "Soon!"})]) == "incomplete_resolution"


def test_tally_and_outputs(tmp_path):
    recs = [attempt("refund_damaged_gold", []), attempt("trap_refund_too_old", [REPLY, REPLY]),
            attempt("trap_refund_too_old", [REPLY])]
    t = tally(recs)
    assert t["failed"] == 2
    assert t["categories"]["no_state_changing_action"] == 1 and t["categories"]["incomplete_resolution"] == 1
    assert "no_state_change" in fail_report(t)
    assert fail_write(t, tmp_path).stat().st_size > 10_000


# ---------- probe comparison ----------
def test_name_grader_matches_state_on_normal_runs():
    task = TASKS["late_gold_update_existing_ticket"]
    rec = run_attempt(task, perfect_for(task, BASE))
    assert name_match_grade(rec, task) is True


def test_name_grader_breaks_under_rename_but_state_does_not():
    task = TASKS["refund_damaged_gold"]
    v = VARIANTS["rename"]
    rec = run_attempt(task, perfect_for(task, v), variant=v)
    assert rec["verified_success"] is True
    assert name_match_grade(rec, task) is False


def test_name_grader_catches_wrong_amount():
    rec = attempt("refund_damaged_gold", [("issue_refund", {"order_id": 101, "amount": 100, "reason": "d"}), REPLY])
    assert name_match_grade(rec, TASKS["refund_damaged_gold"]) is False


def _runs(variant, n=2):
    recs = []
    for task_id in ("refund_damaged_gold", "trap_cancel_shipped", "cancel_processing"):
        for a in range(n):
            r = run_attempt(TASKS[task_id], perfect_for(TASKS[task_id], variant), variant=variant)
            r["attempt"] = a
            recs.append(r)
    return recs


def test_compare_identical_runs_shows_no_effect():
    base = _runs(BASE)
    c = compare(base, json.loads(json.dumps(base)), TASKS)
    assert c["pairs"] == 6
    for g in ("state", "name"):
        assert c[g]["delta"] == 0 and c[g]["flips"] == 0 and not c[g]["effect"]


def test_compare_rename_shows_grader_effect_only(tmp_path):
    base, renamed = _runs(BASE), _runs(VARIANTS["rename"])
    c = compare(base, renamed, TASKS)
    assert c["state"]["delta"] == 0 and not c["state"]["effect"]
    assert c["name"]["delta"] == -1.0 and c["name"]["flips"] == 1.0
    results = {"rename": c}
    assert "EFFECT" in probe_report(results)
    assert probe_write(results, tmp_path).stat().st_size > 10_000
    assert (tmp_path / "probes.csv").exists()


def test_compare_rejects_unmatched_runs():
    a = _runs(BASE, n=1)
    b = json.loads(json.dumps(a))
    for r in b:
        r["attempt"] = 9
    with pytest.raises(ValueError):
        compare(a, b, TASKS)
