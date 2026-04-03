---
name: execution-guardrails
description: Plan-to-execution discipline for large software tasks. Use when Codex needs to keep implementation aligned with the plan, enforce checkpoints, avoid scope drift, monitor progress against milestones, or recover when execution starts deviating from the intended path.
---

# Execution Guardrails

1. Convert plans into checkpoints.
   Each checkpoint should be testable, reviewable, and small enough to verify independently.

2. Define anti-drift signals before coding.
   Watch for:
   - touching files outside the planned scope
   - adding abstractions not required by the current milestone
   - skipping validation
   - solving adjacent problems mid-flight

3. Re-anchor frequently.
   After meaningful edits, restate:
   - what we intended to change
   - what we actually changed
   - whether the delta is still acceptable

4. Prefer thin vertical slices for complex projects.
   Land the minimal end-to-end path first, then expand depth safely.

5. When drift happens, do not hide it.
   Explicitly call out the divergence, explain whether it is justified, and either update the plan or narrow the change set.

6. Finish each execution cycle with evidence.
   Use tests, build output, screenshots, or behavior checks instead of intuition.
