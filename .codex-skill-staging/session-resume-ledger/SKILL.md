---
name: session-resume-ledger
description: Durable progress tracking for long-running coding projects. Use when Codex needs to decompose work, maintain a resume-friendly task ledger, preserve state across interrupted sessions, summarize what changed, or rebuild context quickly after a conversation reset.
---

# Session Resume Ledger

1. Turn broad requests into a compact execution ledger.
   Track:
   - goal
   - current step
   - completed steps
   - blocked items
   - assumptions
   - next actions

2. Prefer resumability over verbosity.
   Write notes so another fresh Codex instance can continue without rereading the full chat.

3. Update checkpoints at natural boundaries.
   Record progress after:
   - a finished edit batch
   - a test run
   - a failed investigation path
   - a design decision

4. Preserve useful negative knowledge.
   Note dead ends, rejected fixes, and misleading symptoms so future sessions do not repeat them.

5. When re-entering a project, reconstruct context in this order:
   - repo status
   - active task
   - last known good state
   - recent edits
   - open risks

6. Keep the final handoff short and action-oriented.
   End with the single best next step plus any exact commands or files worth checking first.
