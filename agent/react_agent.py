"""The AI employee: a simple ReAct-style loop.

    think -> press a button -> see the result -> repeat (max 15 steps)

The loop ends when the model answers without pressing a button. Then the
inspector compares the database before and after.

The model is passed in as a plain function, `model_fn(messages, tools)`, so
tests can use a scripted fake model and the real runs use Ollama.
"""

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from agent.variants import BASE, Variant
from env.db import TODAY, connect, fresh_copy, snapshot
from env.tools import TOOL_SCHEMAS, ShopTools
from eval.checker import check

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
OUTCOME_RE = re.compile(r"OUTCOME:\s*(SUCCESS|FAILED)", re.IGNORECASE)


# ---------- prompt ----------
def system_prompt() -> str:
    return PROMPT_PATH.read_text().format(today=TODAY)


ENV_DIR = Path(__file__).resolve().parent.parent / "env"


def prompt_hash(tools: list[dict] = TOOL_SCHEMAS, extra: str = "") -> str:
    """Fingerprint of the whole setup besides the task: the prompt, the tool
    schemas, the policy, and the tool code. If any of them change, the hash
    changes, so results from different setups never get mixed up by accident.
    `extra` adds a probe variant's settings (empty for the normal setup)."""
    blob = (PROMPT_PATH.read_text() + json.dumps(tools, sort_keys=True)
            + (ENV_DIR / "policy.md").read_text() + (ENV_DIR / "tools.py").read_text() + extra)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------- the real model (Ollama) ----------
class OllamaModel:
    def __init__(self, model: str, temperature: float = 0.7, seed: int = 0, think: bool = False):
        import ollama  # imported here so tests don't need Ollama running

        self.client = ollama.Client()
        self.model = model
        self.options = {"temperature": temperature, "seed": seed}
        self.think = think

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.chat(
            model=self.model, messages=messages, tools=tools,
            think=self.think, options=self.options,
        )
        return parse_response(resp)


def parse_response(resp) -> dict:
    """Turn an Ollama ChatResponse into a plain dict the loop understands."""
    msg = resp.message
    calls = []
    for tc in msg.tool_calls or []:
        args = tc.function.arguments
        if isinstance(args, str):  # some models send arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"name": tc.function.name, "arguments": dict(args or {})})
    return {
        "content": msg.content or "",
        "tool_calls": calls,
        "prompt_tokens": resp.prompt_eval_count or 0,
        "completion_tokens": resp.eval_count or 0,
    }


# ---------- one attempt ----------
def run_attempt(task: dict, model_fn, *, max_steps: int = 15, variant: Variant = BASE) -> dict:
    """Run one task once, on a fresh copy of the shop, and grade it.

    `variant` controls only what the agent is shown (tool names, result
    format). The real tools and the checker never change.
    """
    workdir = Path(tempfile.mkdtemp(prefix="arb_"))
    conn = connect(fresh_copy(workdir))
    tools = ShopTools(conn)
    tools_schema = variant.schemas()

    messages = [
        {"role": "system", "content": variant.rename_text(system_prompt())},
        {"role": "user", "content": task["user_message"]},
    ]
    tool_log, prompt_tokens, completion_tokens = [], 0, 0
    finish, error, n_steps = "max_steps", None, 0
    start = time.time()
    before = snapshot(conn)

    try:
        for n_steps in range(1, max_steps + 1):
            out = model_fn(messages, tools_schema)
            prompt_tokens += out["prompt_tokens"]
            completion_tokens += out["completion_tokens"]

            assistant = {"role": "assistant", "content": out["content"]}
            if out["tool_calls"]:
                assistant["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in out["tool_calls"]
                ]
            messages.append(assistant)

            if not out["tool_calls"]:
                finish = "stop"
                break

            for c in out["tool_calls"]:
                real_name = variant.real_name(c["name"])
                if real_name is None:
                    result = {"error": f"Unknown tool '{c['name']}'"}
                else:
                    result = tools.call(real_name, c["arguments"])
                tool_log.append({"step": n_steps, "name": c["name"], "real_name": real_name,
                                 "args": c["arguments"], "result": result})
                messages.append({"role": "tool", "tool_name": c["name"],
                                 "content": variant.format_result(result)})
    except Exception as e:  # the model server crashed / timed out: not the agent's fault
        finish, error = "infra_error", f"{type(e).__name__}: {e}"

    after = snapshot(conn)
    conn.close()
    shutil.rmtree(workdir, ignore_errors=True)

    verdict = check(task, before, after)
    texts = [m["content"] for m in messages if m["role"] == "assistant" and m["content"]]
    tags = [t.upper() for text in texts for t in OUTCOME_RE.findall(text)]
    outcome_tag = tags[-1] if tags else None
    # "Claimed success": the agent said SUCCESS, or stopped on its own without saying FAILED.
    claimed = outcome_tag == "SUCCESS" or (outcome_tag is None and finish == "stop")

    return {
        "task_id": task["id"],
        "difficulty": task.get("difficulty"),
        "category": task.get("category"),
        "finish_reason": finish,
        "error": error,
        "steps": n_steps,
        "seconds": round(time.time() - start, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tool_calls": tool_log,
        "tool_errors": sum("error" in t["result"] for t in tool_log),
        "final_text": texts[-1] if texts else "",
        "outcome_tag": outcome_tag,
        "claimed_success": claimed,
        "verified_success": verdict["passed"],
        "checker": {k: verdict[k] for k in ("missing", "forbidden_hit", "unexpected",
                                             "no_state_change", "diff")},
        "messages": messages,
    }