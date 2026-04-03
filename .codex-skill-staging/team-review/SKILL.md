---
name: team-review
description: Professional code review workflow aligned with real engineering teams. Use when Codex needs to review code for correctness, regressions, maintainability, missing tests, rollout risk, unclear ownership boundaries, or operational impact rather than giving generic style feedback.
---

# Team Review

1. Review for risk before style.
   Prioritize:
   - correctness
   - behavioral regression
   - data integrity
   - security
   - operability
   - test coverage gaps

2. Read changes in execution context.
   Understand surrounding flows, not just the diff hunk.

3. Comment like a teammate.
   Findings should be:
   - specific
   - actionable
   - severity-aware
   - respectful

4. Always look for missing validation.
   Call out absent tests, weak assertions, incomplete migrations, and silent failure paths.

5. Examine production realism.
   Ask whether the change behaves correctly under:
   - retries
   - partial failure
   - concurrency
   - stale clients
   - backward compatibility pressure

6. Keep the summary secondary.
   Lead with findings first. If there are no findings, say so clearly and mention residual risks.
