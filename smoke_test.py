import time
import ollama

MODEL = "qwen3.5:4b-mlx"   
RUNS = 4


def get_order(order_id: int) -> dict:
    """Look up an order by its ID.

    Args:
        order_id: The order's numeric ID
    """
    return {"id": order_id, "status": "shipped", "eta": "2026-10-28"}


passed = 0

for i in range(1, RUNS + 1):
    start = time.time()

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Where is my order 42?"}],
        tools=[get_order],
        think=False,              # turn off "thinking" mode for speed
        options={"temperature": 0},
    )

    seconds = time.time() - start
    calls = resp.message.tool_calls

    ok = (
        calls is not None
        and len(calls) > 0
        and calls[0].function.name == "get_order"
        and int(calls[0].function.arguments.get("order_id", -1)) == 42
    )

    if ok:
        passed += 1
        print(f"Run {i}: PASS  ({seconds:.1f}s)  {calls[0].function.name}({calls[0].function.arguments})")
    else:
        print(f"Run {i}: FAIL  ({seconds:.1f}s)")
        print(f"   tool_calls: {calls}")
        print(f"   text reply: {resp.message.content!r}")

print(f"\n{MODEL}: {passed}/{RUNS} runs pressed the right button with the right argument")