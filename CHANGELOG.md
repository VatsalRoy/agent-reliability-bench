# Changelog

Changes to the benchmark setup that affect results. Each change bumps the
setup fingerprint (`prompt_hash` in every run log), so old and new runs can't
be mixed by accident.

## v2 (after baseline v1): fixed two harness bugs

Baseline v1 (qwen3.5:4b-mlx, k=1): agent claimed success 17/17, database
confirmed 10/17. Reading all 7 failures showed that 3 were partly caused by
the setup, not the agent:

1. **`search_policy` returned only the top 3 sections.** For refund queries,
   the "Large refunds (manager approval)" and "Tickets" sections never
   appeared, so the agent could not follow rules it was never shown
   (`trap_large_refund_on_hold`). Fix: return every matching section, best first.
2. **The Tickets rule was ambiguous.** It said to mark a ticket solved "only
   when" the issue is resolved: a restriction, not an instruction. Two tasks
   (`late_gold_update_existing_ticket`, `multi_two_issues`) required closing
   the ticket. Fix: the rule now says to set the existing ticket to "solved"
   when you resolve the order's issue.

**Not changed:** the agent's system prompt, the tasks, and the checker. The
other 4 failures were genuine agent errors and are left as they are:
answering in chat instead of calling `send_reply` (3 lookups), and
mis-computing the 30-day refund window (`trap_refund_too_old`).

Lesson (matches TRACE, arXiv 2610.11678): read failures before trusting a
score. A failure can come from the harness rather than the agent.