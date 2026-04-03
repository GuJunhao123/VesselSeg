---
name: architecture-decision-playbook
description: Architecture selection and system design decision support for complex backend-heavy or full-stack projects. Use when Codex needs to compare architecture patterns, choose service boundaries, evaluate tradeoffs, identify likely scaling or coupling risks, write ADR-style reasoning, or prevent common architecture mistakes before implementation.
---

# Architecture Decision Playbook

1. Start from constraints, not patterns.
   Capture traffic shape, consistency needs, team size, deployment model, failure tolerance, data ownership, and delivery pressure.

2. Compare at least 2 viable options.
   Include a conservative option and an aggressively scalable option. Reject options explicitly when they add coordination or operational cost without enough payoff.

3. Make tradeoffs concrete.
   Always discuss:
   - complexity
   - deployability
   - observability
   - data consistency
   - testability
   - team cognitive load
   - migration cost

4. Prefer staged evolution.
   Recommend the simplest architecture that can survive the next realistic growth phase. Call out what should stay modular now so it can split later.

5. Produce an ADR-style answer when the task is open-ended.
   Include:
   - context
   - decision
   - alternatives considered
   - consequences
   - follow-up checks

6. Flag common traps early.
   Watch for:
   - microservices before clear domain boundaries
   - premature event-driven complexity
   - shared database across services
   - state management scattered across frontend routes
   - over-custom component systems without design tokens
   - missing ownership boundaries for cross-cutting concerns

7. End with implementation guidance.
   Translate the design into the next practical steps, such as repo structure, module seams, API boundaries, migration order, and observability checkpoints.
