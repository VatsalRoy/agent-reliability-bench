# Literature notes

One paragraph per source: what they did, the key numbers, and what this project borrows. Numbers marked ✓ were checked against the paper's abstract or summary on Oct 29, 2026.

## ThinkingBox (Li et al., arXiv 2608.19741; Microsoft × Hugging Face, Oct 2026)

**What:** A sandbox and benchmark of 507 simulated business workflows ✓, each run 20 times ✓, graded by the final database state rather than the transcript ✓.
**Key numbers:** pass@1 overstates dependable performance. For example, Qwen3.8-27B reaches 51.70% pass@1 but only 10.93% pass^20 (and 89.35% pass@20) ✓, and Claude Opus 5 falls from 66.50% to 47.53% ✓. Failure categories averaged over models: tool usage 79.9%, wrong state update 10.3%, incomplete user resolution 7.0%, no state-changing action 2.9% ✓. Their taxonomy has no "reasoning" category ✓.
**Borrowed:** state-based grading, repeated trials, pass@1 / pass@k / pass^k, cost per dependable task, and the four failure categories.

## TRACE: Diagnosing Verifier Brittleness in Agentic Evaluation (Gaonkar, arXiv 2610.11678, Oct 2026)

**What:** A protocol for testing whether a score change comes from the agent or from the evaluation: change one part of the evaluation (paired mutation), check whether behavior changed, and rescore unchanged trajectories ✓. Studies: a 25-task synthetic suite, 30 τ²-bench tasks with four agents, and an 88-task follow-up with repeats ✓.
**Key numbers:** renaming tools lowers a scripted agent's score by 0.250 with identical operations ✓; rename or reformat stays within ±0.10 for 7 of 8 agent-change pairs ✓; misleading names lower reward by 0.20–0.44 ✓; identical reruns flip 15–36% of outcomes ✓; two frontier LLM judges disagree on 57% of records ✓.
**Borrowed:** the rename/reformat probes, the repeat control, the misleading-name positive control, and rescoring the same trajectories with a second grader.

## Agents Are Systems, Not Models (arXiv 2610.01618, Oct 2026)

**Not yet verified.** I couldn't retrieve this paper while writing these notes. The project plan summarizes it as: about 54% of outcome variance comes from re-running the same configuration; the information given to the agent matters more than model size; and a verification *tool* changes behavior where verification *prompting* doesn't. **Read the paper and confirm these before citing them.**
**Borrowed:** the motivation for the identical-rerun control, and the proposed `verify_state` ablation.

## Gap this project fills

These studies use large benchmarks and frontier models. This project tests whether the same failure modes appear in a small, cheap, reproducible setup that a team could run in CI. They mostly do: repeated runs expose a large reliability gap, judges are lenient, and presentation changes move naive graders but not state checks. The one exception is the failure mix, which here is dominated by silence and policy errors rather than tool-usage errors.
