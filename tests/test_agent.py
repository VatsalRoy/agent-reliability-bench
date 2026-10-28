"""Tests for the agent loop, using scripted fake models instead of Ollama.
They prove the plumbing works, so any failure in a real run is the model's."""

import json

import pytest
from ollama import ChatResponse, Message

from agent.react_agent import parse_response, prompt_hash, run_attempt
from agent.run_bench import run, summarize
from eval.checker import load_tasks

TASKS = {t["id"]: t for t in load_tasks()}


def out(content="", calls=()):
    return {"content": content, "prompt_tokens": 100, "completion_tokens": 20,
            "tool_calls": [{"name": n, "arguments": a} for n, a in calls]}


class Scripted:
    """A fake model that returns pre-written responses, one per step."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def perfect_model(task):
    """Replays the task's reference solution, one tool call per step."""
    steps = [out(calls=[(s["tool"], s["args"])]) for s in task["reference_solution"]]
    return Scripted(steps + [out("All handled.\nOUTCOME: SUCCESS")])


# ---------- the loop solves every task when the model is perfect ----------
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_perfect_model_passes(task_id):
    rec = run_attempt(TASKS[task_id], perfect_model(TASKS[task_id]))
    assert rec["finish_reason"] == "stop"
    assert rec["verified_success"] and rec["claimed_success"]
    assert rec["tool_errors"] == 0


# ---------- the headline failure: says done, did nothing ----------
def test_liar_is_caught():
    model = Scripted([
        out(calls=[("send_reply", {"message": "Your refund of $120 has been issued!"})]),
        out("OUTCOME: SUCCESS"),
    ])
    rec = run_attempt(TASKS["refund_damaged_gold"], model)
    assert rec["claimed_success"] is True
    assert rec["verified_success"] is False
    assert rec["checker"]["missing"][0]["table"] == "refunds"


def test_outcome_failed_is_not_a_claim():
    rec = run_attempt(TASKS["refund_damaged_gold"], Scripted([out("I can't do this.\nOUTCOME: FAILED")]))
    assert rec["outcome_tag"] == "FAILED" and rec["claimed_success"] is False


def test_no_tag_but_stopped_counts_as_claim():
    rec = run_attempt(TASKS["refund_damaged_gold"], Scripted([out("All done!")]))
    assert rec["outcome_tag"] is None and rec["claimed_success"] is True


# ---------- loop mechanics ----------
def test_step_limit():
    model = Scripted([out(calls=[("get_order", {"order_id": 101})])] * 5)
    rec = run_attempt(TASKS["refund_damaged_gold"], model, max_steps=5)
    assert rec["finish_reason"] == "max_steps" and rec["steps"] == 5
    assert rec["claimed_success"] is False


def test_tool_errors_are_fed_back_to_model():
    model = Scripted([
        out(calls=[("refund_order", {"order_id": 101})]),  # wrong tool name
        out(calls=[("issue_refund", {"order_id": 101, "amount": 120, "reason": "damaged"})]),
        out(calls=[("send_reply", {"message": "Refunded."})]),
        out("OUTCOME: SUCCESS"),
    ])
    rec = run_attempt(TASKS["refund_damaged_gold"], model)
    assert rec["tool_errors"] == 1 and rec["verified_success"]
    error_msg = model.seen[1][-1]
    assert error_msg["role"] == "tool" and "Unknown tool" in error_msg["content"]


def test_infra_error_is_separated():
    def broken(messages, tools):
        raise ConnectionError("Ollama is not running")

    rec = run_attempt(TASKS["refund_damaged_gold"], broken)
    assert rec["finish_reason"] == "infra_error" and "Ollama" in rec["error"]


def test_attempts_do_not_share_state():
    """Two refunds on separate attempts must not see each other."""
    task = TASKS["refund_damaged_gold"]
    a = run_attempt(task, perfect_model(task))
    b = run_attempt(task, perfect_model(task))
    assert a["verified_success"] and b["verified_success"]


def test_record_is_json_serializable():
    rec = run_attempt(TASKS["multi_two_issues"], perfect_model(TASKS["multi_two_issues"]))
    json.dumps(rec)


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 12


# ---------- parsing real Ollama responses ----------
def test_parse_ollama_response():
    resp = ChatResponse(
        message=Message(role="assistant", content="",
                        tool_calls=[Message.ToolCall(function=Message.ToolCall.Function(
                            name="get_order", arguments={"order_id": 42}))]),
        prompt_eval_count=50, eval_count=10,
    )
    parsed = parse_response(resp)
    assert parsed["tool_calls"] == [{"name": "get_order", "arguments": {"order_id": 42}}]
    assert parsed["prompt_tokens"] == 50 and parsed["completion_tokens"] == 10


def test_parse_plain_text_response():
    resp = ChatResponse(message=Message(role="assistant", content="Hello"))
    parsed = parse_response(resp)
    assert parsed["tool_calls"] == [] and parsed["content"] == "Hello"
    assert parsed["prompt_tokens"] == 0


# ---------- the runner ----------
def test_runner_writes_jsonl(tmp_path):
    tasks = [TASKS["refund_damaged_gold"], TASKS["trap_cancel_shipped"]]
    by_id = {t["id"]: t for t in tasks}
    queue = [t["id"] for t in tasks for _ in range(2)]  # k=2, in run order

    def make_model(seed):
        return perfect_model(by_id[queue.pop(0)])

    out_path = tmp_path / "runs" / "test.jsonl"
    records = run(tasks, make_model, k=2, out_path=out_path, meta={"model": "fake"}, quiet=True)
    lines = out_path.read_text().splitlines()
    assert len(lines) == 4 == len(records)
    first = json.loads(lines[0])
    assert first["model"] == "fake" and first["attempt"] == 0 and first["seed"] == 0
    assert "100%" in summarize(records)