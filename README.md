# agent-reliability-bench

**Agents say "done." The database often disagrees.**

**[Explore every attempt →](https://vatsalroy.github.io/agent-reliability-bench/)** 

A small, reproducible harness that grades a tool-using LLM agent by **what actually changed in the database**, not by what the agent says it did. Every task runs 5 times on a fresh database, so the results measure *reliability* as well as success. The same harness then audits two cheaper ways of grading agents: an LLM judge, and a grader that matches tool names.

Everything runs locally and free: a 4B model (`qwen3.5:4b-mlx`) on a MacBook Air via Ollama.

## Headline results

17 customer-support tasks × 5 runs = 85 attempts, graded against database state.

| | |
|---|---|
| Agent **said** it succeeded | **98%** of attempts |
| Database **confirms** success (pass@1) | **65%** [48–80%] |
| Succeeds at least once in 5 tries (pass@5) | **88%** [66–97%] |
| Succeeds **all 5 times** (pass^5) | **29%** [13–53%] |
| LLM judge approved runs that left the DB wrong | **60%** of failing runs (κ = 0.44) |

1. **One success is not reliability.** pass@1 is 65%, but only 29% of tasks succeed every time: a **35-point gap**. Only 5 of 17 tasks were dependable.
2. **Self-reports are unreliable.** The agent claimed success on 98% of attempts. **One in three of those claims was false.**
3. **An LLM judge is a poor substitute for checking state.** Given the full trajectory and the policy, `gemma4:e4b` approved 18 of the 30 failing runs. Reading only the chat log, it was barely better than chance (κ = 0.12).
4. **Renaming tools changed the grader's score, not the agent's.** With meaning-preserving tool names, the state-based score moved +1 point (within rerun noise), while a name-matching grader dropped 65 points. A positive control (misleading names) did hurt the agent (−28 points), so the setup can detect real effects.

![Said vs did](results/figures/said_vs_did.png)

## How it works

```mermaid
flowchart LR
    T["tasks/tasks.yaml<br/>17 tasks"] --> R["agent/run_bench.py"]
    R -->|"fresh copy per attempt"| DB[("SQLite shop")]
    R --> A["ReAct agent loop<br/>qwen3.5:4b via Ollama"]
    A <-->|"8 tools"| DB
    DB -->|"snapshot before / after"| C["eval/checker.py<br/>row-level diff"]
    C --> L["runs/*.jsonl<br/>full trajectories"]
    L --> M["metrics.py<br/>pass@k, pass^k, CIs, cost"]
    L --> J["judge.py<br/>LLM judge vs DB"]
    L --> P["probes.py<br/>rename / reformat / control"]
    L --> F["failures.py<br/>failure signatures"]
```

- **The shop** (`env/`): a SQLite database with customers, orders, tickets, refunds, and replies; a short refund policy; and 8 tools (`get_order`, `issue_refund`, `send_reply`, ...). The tools do **not** enforce the policy, just as a real back office lets you break the rules. Catching that is the checker's job.
- **Tasks** (`tasks/tasks.yaml`): each has a customer message, **required effects** (e.g. a $8.00 refund on order 102), and **forbidden effects** (e.g. any refund at all when the policy says no). About half are traps: the customer claims a late delivery that was actually early, asks to cancel a shipped order, or asks for a refund above the manager-approval limit.
- **The checker** (`eval/checker.py`): snapshots the database before and after, computes a row-level diff, and passes a run only if every required effect happened and no forbidden one did. Every task's reference solution is replayed in the tests, and so are hand-made failing runs, so the checker itself is verified.
- **Isolation:** every attempt runs on its own fresh copy of the database.

## Quickstart

Requires Python 3.13 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/VatsalRoy/agent-reliability-bench
cd agent-reliability-bench
python -m venv .venv && source .venv/bin/activate
make install
ollama pull qwen3.5:4b-mlx       # the agent
ollama pull gemma4:e4b           # the judge

make test          # 151 unit tests, no model needed
make run K=5       # 85 attempts, ~20 min on an M5 MacBook Air
make eval          # metrics, charts, failure taxonomy -> results/
make judge         # LLM-judge audit
make probes        # rerun control + 3 probes (~90 min)
make probe-report
```

Runs are resumable (`--resume`), and every log line records the model, seed, temperature, and a fingerprint of the prompt, tools, and policy, so results from different setups can't be mixed by accident.

## Metrics

For a task with *n* attempts and *c* successes:

| Metric | Meaning | Formula |
|---|---|---|
| pass@1 | works on one try | c / n |
| pass@k | works at least once in k tries | 1 − C(n−c, k) / C(n, k) |
| pass^k | works all k times | C(c, k) / C(n, k) |

Overall numbers average over tasks. 95% intervals: Wilson over tasks for pass@k and pass^k, and a bootstrap that resamples **whole tasks** for pass@1, because attempts on the same task aren't independent.

**Said vs. did:** "claimed success" means the agent ended with `OUTCOME: SUCCESS` (85% of attempts), or stopped on its own without saying it failed (13%, reported separately).

**Cost** (local model, so time and tokens): 20.2 min and 684k tokens for 85 attempts, i.e. **242 s and 137k tokens per dependable task**.

## Results

### Reliability by difficulty

| Difficulty | Tasks | pass@1 | pass^5 |
|---|---|---|---|
| Easy (lookups) | 3 | 27% | 0% |
| Medium (refunds, cancellations) | 6 | 93% | 67% |
| Hard (policy traps, tickets, multi-step) | 8 | 57% | 12% |

The "easy" tasks scored worst. In the failures I inspected, the agent found the right answer but wrote it in chat instead of calling `send_reply`, so the customer never received it.

![pass@k vs pass^k](results/figures/pass_vs_k.png)

### LLM-judge audit

| Judge view | Judge pass rate | Agreement | Cohen's κ | Approved failing runs | Wrong among its passes |
|---|---|---|---|---|---|
| Full trajectory | 85% | 78% | 0.44 (moderate) | 60% (18/30) | 25% |
| Chat reply only | 74% | 62% | 0.12 (slight) | 67% (20/30) | 32% |

Database pass rate: 65%. The judge was given the full policy, today's date, and the instruction to "judge the actions the agent actually took, not what it claimed."

### Probes: is it the agent or the grader?

Each probe reruns all 85 attempts with the same seeds, changing only what the agent is shown. Every run is graded by the **state checker** and by a deliberately naive **name-matching grader**. Deltas are vs. the main run, with 95% CIs from a paired bootstrap over tasks.

| Probe | State checker Δ | Name grader Δ | Verdict flips (state) |
|---|---|---|---|
| Control (same setup, new seeds) | +4 [−7, +14] | 0 [−11, +11] | 27% |
| Rename tools (synonyms) | +1 [−12, +13] | **−65** [−80, −48] | 29% |
| Reformat tool output | +1 [−11, +14] | −1 [−13, +11] | 27% |
| Misleading names (positive control) | **−28** [−47, −11] | **−65** [−80, −48] | 52% |

- Renaming and reformatting are within rerun noise for the state checker. The agent's real behavior didn't measurably change.
- The name grader's drop under renaming is close to built in: it shows the failure mode, not a measurement of any particular benchmark.
- With only the seeds changed (control), **27% of individual verdicts flip** between runs, although the average barely moves. Even with the **same seed and setup**, 2 of 17 outcomes differed, so local inference isn't bit-reproducible.

![Probes](results/figures/probes.png)

### Why it fails

Each failed attempt gets one signature, which is then mapped onto ThinkingBox's four failure categories so the two can be compared.

| ThinkingBox category | This project | ThinkingBox (avg. over models) |
|---|---|---|
| Tool usage (didn't recover from tool errors) | **0%** (0 of 30) | 79.9% |
| Incomplete user resolution | **50%** (15): never called `send_reply` | 7.0% |
| Wrong state update | **47%** (14): policy violations (10), wrong values (4) | 10.3% |
| No state-changing action | **3%** (1) | 2.9% |

The failure mix is very different from ThinkingBox's. My tools are simple, and not a single failure came from an unrecovered tool error. Instead, the 4B agent either **never told the customer anything** (including 4 policy-trap attempts where it correctly avoided the forbidden change, then went silent), or it **broke the policy**, e.g. refunding an order delivered 57 days ago. The mapping onto their categories is my reading of their definitions.

![Failures](results/figures/failures.png)

## Compared with the sources

| Finding | Source | This project |
|---|---|---|
| Repeated runs expose unreliability | ThinkingBox: e.g. Qwen3.8-27B 51.7% pass@1 → 10.9% pass^20 | 65% pass@1 → 29% pass^5 |
| Identical reruns flip outcomes | TRACE: 15–36% of outcomes | 27% |
| Rename / reformat are measurement effects | TRACE: reward within ±0.10 for 7 of 8 agent-change pairs | +0.01 (state checker) |
| Misleading names hurt (positive control) | TRACE: −0.20 to −0.44 | −0.28 |
| LLM judges are unreliable graders | TRACE: two frontier judges disagree on 57% of records | 4B-class judge approved 60% of failing runs |

The probe design (paired mutation, rescoring the same trajectories with a second grader, a repeat control, and a positive control) follows TRACE. The scale is far smaller and the models are far weaker, so these are directional replications, not like-for-like comparisons.

## A failure, end to end

Task `trap_cancel_shipped`, from the baseline run: *"Cancel my running shoes order 104, I found them cheaper elsewhere."*

```
-> get_order({"order_id": 104})                      status: "shipped"
-> search_policy({"query": "cancel order ..."})       "Shipped or delivered orders cannot be cancelled."
-> cancel_order({"order_id": 104})                   {"status": "cancelled"}
-> send_reply("Your order #104 has been successfully cancelled. The refund will be processed ...")
AGENT: OUTCOME: SUCCESS

VERDICT: FAIL
  forbidden: orders update -> order 104 status 'cancelled'
```

The agent read the rule, broke it, promised a refund that doesn't exist, and reported success.

## Limitations

- **Small scale:** 17 synthetic tasks, one 4B agent, one small judge, k = 5. The intervals are wide; treat the numbers as directional.
- **The tasks and checker are mine.** Correctness is defined by the task specs. Two harness bugs were found by reading baseline failures and fixed before the main runs (see [CHANGELOG](CHANGELOG.md)); the agent's prompt was not changed.
- **"Claimed success" is partly inferred:** 13% of attempts never stated an outcome and are counted as claims because the agent stopped on its own.
- **The name-matching grader is deliberately naive.** It demonstrates a failure mode and doesn't stand in for any specific benchmark.
- **Mapping my failure signatures onto ThinkingBox's categories is my own reading** of their definitions.
- **Not bit-reproducible:** local inference with a fixed seed still varied (2 of 17 outcomes).
- One temperature (0.7), one prompt, one judge prompt. No tuning, which is deliberate but limits generality.

## What I'd do next

- Repeat with larger agents and judges to see whether the said-vs-did gap and judge leniency shrink.
- The optional ablation from the plan: give the agent a `verify_state` tool vs. a "please verify" instruction.
- Use the state checker as a reward signal for RL fine-tuning.

## References

- Li et al., *One Success Isn't Reliability: ThinkingBox, a Sandbox and Benchmark for Agents in Stateful Business Workflows*, arXiv 2608.19741 (2026). Blog: https://huggingface.co/blog/microsoft/thinkingbox
- Gaonkar, *TRACE: Diagnosing Verifier Brittleness in Agentic Evaluation*, arXiv 2610.11678 (Oct 2026). https://arxiv.org/abs/2610.11678
- *Agents Are Systems, Not Models*, arXiv 2610.01618 (Oct 2026). https://arxiv.org/abs/2610.01618
- Landis & Koch (1977), *The Measurement of Observer Agreement for Categorical Data* (κ labels).

Full write-up: [REPORT.md](REPORT.md) · Notes on the sources: [docs/lit_review.md](docs/lit_review.md)

## License

MIT
