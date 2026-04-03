---
name: bug-investigation-playbook
description: Structured debugging workflow for difficult or strange software bugs. Use when Codex needs to triage an unclear failure, reduce a large search space, form and test hypotheses systematically, correlate logs and code paths, or avoid random walk debugging.
---

# Bug Investigation Playbook

1. Stabilize the symptom.
   Define:
   - observed behavior
   - expected behavior
   - reproduction frequency
   - last known good state

2. Narrow the search space before patching.
   Slice by:
   - layer
   - environment
   - recent changes
   - input shape
   - timing or concurrency

3. Work hypothesis-first.
   For each hypothesis, record:
   - why it fits
   - the cheapest falsifying check
   - the evidence outcome

4. Prefer instrumentation over guesswork.
   Add targeted logs, assertions, probes, or temporary guards to reveal state transitions.

5. Keep an investigation log.
   Preserve:
   - confirmed facts
   - disproven ideas
   - suspicious code paths
   - remaining unknowns

6. Do not ship the first plausible fix.
   Verify root cause, test nearby failure modes, and state residual risk explicitly.

7. For frontend or full-stack bugs, inspect all of:
   - route transitions
   - async state updates
   - stale cache behavior
   - responsive breakpoints
   - API contract mismatches
