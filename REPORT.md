# The Agent Said Done: Measuring What a Small Tool-Using Agent Actually Did

**Vatsal Roy** · October 2026 · Code and data: https://github.com/VatsalRoy/agent-reliability-bench

## Abstract

Agent evaluations often report a single success rate, judged from the agent's transcript. I built a small, reproducible harness that instead grades a tool-using agent by the **final state of a database**, runs every task five times on a fresh copy, and uses that ground truth to audit two cheaper graders. On 17 customer-support tasks, a local 4B model (`qwen3.5:4b-mlx`) claimed success on 98% of attempts, but the database confirmed only 65%. Only 29% of tasks succeeded on all five tries. An LLM judge (`gemma4:e4b`) approved 60% of the runs that left the database wrong (κ = 0.44), falling to κ = 0.12 when it saw only the chat. TRACE-style probes showed that renaming tools changed a name-matching grader's score by −65 points while leaving the state-based score within rerun noise (+1), and a misleading-name positive control reduced the state-based score by 28 points. These results reproduce, at small scale and with weak models, the main findings of ThinkingBox and TRACE.

## 1. Question

How much do (a) single-run scores and (b) LLM-as-judge grading overstate a small agent's reliability, compared with repeated, state-based executable checks?

Hypotheses (written in the project plan before any runs):

- **H1.** pass^k (succeeds every time) is much lower than pass@1.
- **H2.** An LLM judge that reads the trajectory passes a meaningful share of runs that fail the state checks.
- **H3.** Meaning-preserving tool renames don't change state-based scores beyond rerun noise, but they do change a name-matching grader.

## 2. Setup

**Environment.** A SQLite "shop" with customers, orders, tickets, refunds, and replies, plus a one-page refund policy (tier-based partial refunds for late delivery, a 30-day window, a $500 manager-approval limit, cancellation only while processing). The agent acts only through 8 tools. The tools enforce data integrity but **not** the policy.

**Tasks.** 17 tasks: 3 lookups, 6 refunds/cancellations, 5 "policy says no" traps, and 3 ticket and multi-step tasks. Each specifies **required effects** (e.g. exactly one refund of $8.00 on order 102) and **forbidden effects** (e.g. any refund). A run passes only if every required effect occurred and no forbidden effect did.

**Checker validation.** Every task's reference solution is replayed in the unit tests and must pass, and doing nothing must fail. Nine hand-made wrong runs (wrong amount, two replies, ticket solved instead of put on hold, ...) must fail. The suite has 151 tests in total.

**Agent.** A plain ReAct tool-calling loop (max 15 steps) over `qwen3.5:4b-mlx` via Ollama on a MacBook Air (M5, 16 GB), temperature 0.7, seed = attempt index, thinking mode off. Every attempt runs on a fresh copy of the database. The agent ends with `OUTCOME: SUCCESS` or `OUTCOME: FAILED`, which defines its self-report.

**Harness corrections.** Reading the first baseline's failures revealed two harness bugs: the policy search returned only the top 3 sections, which hid the manager-approval rule, and the ticket rule was worded ambiguously. Both were fixed before the main experiments. The agent's prompt, the tasks, and the checker were not changed. See `CHANGELOG.md`.

## 3. Results

### 3.1 Reliability (H1)

| Metric | Value | 95% CI |
|---|---|---|
| pass@1 | 65% | 48–80% (task bootstrap) |
| pass@5 | 88% | 66–97% (Wilson) |
| pass^5 | 29% | 13–53% (Wilson) |

The 35-point gap between pass@1 and pass^5 supports H1. Medium tasks were most dependable (pass@1 93%, pass^5 67%). Lookups were least dependable (27%, 0%), because the agent answered in chat without calling `send_reply`. Cost per dependable task was 242 s and 137k tokens.

### 3.2 Said vs. did

The agent claimed success on 98% of attempts: 85% explicitly, and 13% by stopping without stating an outcome. Of all its success claims, 34% were false.

### 3.3 LLM judge audit (H2)

| View | Judge pass rate | κ | Approved failing runs | Wrong among its passes |
|---|---|---|---|---|
| Full trajectory | 85% | 0.44 | 60% (18/30) | 25% |
| Reply only | 74% | 0.12 | 67% (20/30) | 32% |

H2 is supported. Even with the full trajectory, the policy, and an instruction to judge actions rather than claims, the judge approved most failing runs. Several of these were lookups where the agent never called `send_reply`: the answer appeared in the agent's own notes, and the judge treated it as delivered. With only the chat, the judge was both too lenient and too strict, failing 12 correct runs whose actions it couldn't see.

### 3.4 Probes (H3)

All probes reuse the main run's seeds and change only what the agent sees. Deltas are in pass@1 points with paired task-bootstrap 95% CIs.

| Probe | State checker | Name-matching grader | Outcome flips |
|---|---|---|---|
| Control (new seeds) | +4 [−7, +14] | 0 [−11, +11] | 27% |
| Rename (synonyms) | +1 [−12, +13] | −65 [−80, −48] | 29% |
| Reformat (pretty JSON) | +1 [−11, +14] | −1 [−13, +11] | 27% |
| Misleading names | −28 [−47, −11] | −65 [−80, −48] | 52% |

H3 is supported: renaming moved the name grader by 65 points and the state checker by +1, within noise. The positive control shows the setup can detect a real effect. A caveat: the name grader's collapse under renaming is nearly guaranteed by its construction. It shows the failure mode TRACE warns about; it does not measure a particular benchmark. The more informative result is that the state checker stayed stable while the presentation changed.

**Noise.** Changing only the seeds flipped 27% of individual verdicts while moving the average by 4 points. With the same seed and setup, 2 of 17 outcomes still differed, so local inference was not bit-reproducible.

### 3.5 Failure taxonomy

Of 30 failed attempts, mapped onto ThinkingBox's categories: incomplete user resolution 50% (never called `send_reply`), wrong state update 47% (10 policy violations, 4 wrong values), no state-changing action 3%, and tool usage 0%. ThinkingBox reports tool usage as 79.9% of failures. Here the tools are simple, and the small agent's failures were silence and policy errors, not unrecovered tool errors.

## 4. Relation to prior work

The design follows **ThinkingBox** (Li et al., arXiv 2608.19741): grading by terminal database state, and reporting pass@1, pass@k, and pass^k over repeated trials. The probes follow **TRACE** (Gaonkar, arXiv 2610.11678): paired mutation, rescoring the same trajectories with a second grader, a repeat control, and a positive control. My numbers fall inside TRACE's reported ranges: rerun flips of 15–36% (here 27%), misleading names costing 0.20–0.44 (here 0.28), and rename/reformat staying within ±0.10 for most agent-change pairs (here +0.01).

## 5. Limitations

17 synthetic tasks written by the author; one small agent model and one small judge; k = 5; wide intervals. Correctness is defined by my task specs. "Claimed success" is partly inferred. The name-matching grader is deliberately naive. The category mapping onto ThinkingBox's taxonomy is my interpretation. Local inference was not bit-reproducible. One temperature and one prompt were used, with no tuning.

## 6. Takeaways

For anyone shipping agents: **grade side effects, not transcripts; repeat runs and report consistency (pass^k); and validate any cheaper grader against state before trusting it.** Natural next steps are larger models, a `verify_state` tool ablation, and using the state checker as a training reward.
