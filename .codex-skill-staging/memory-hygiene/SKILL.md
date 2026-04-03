---
name: memory-hygiene
description: Memory curation and cleanup for Codex project knowledge. Use when Codex needs to capture durable project facts, separate stable knowledge from temporary debugging notes, prune stale memories, or reduce noisy context that harms future responses.
---

# Memory Hygiene

1. Classify every candidate memory before keeping it.
   Use these buckets:
   - durable fact
   - temporary hypothesis
   - obsolete note
   - workflow preference

2. Keep only durable, reusable knowledge by default.
   Good memories include:
   - architecture decisions
   - repo conventions
   - deployment constraints
   - naming rules
   - recurring gotchas that are still true

3. Delete or avoid storing low-value noise.
   Examples:
   - one-off stack traces
   - outdated TODOs
   - superseded debugging guesses
   - transient branch names

4. Compress instead of accreting.
   Merge overlapping memories into one cleaner note when possible.

5. Tag risk of staleness mentally.
   If a fact may age quickly, say so and prefer source-of-truth files over memory.

6. When asked to summarize history, separate:
   - what is still true
   - what was tried
   - what should be forgotten
