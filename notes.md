# Ablation Ideas (future work)

## C3 dual-stream: self vs. global stream ablation

Current C3 (`src/data.py:236-305` `IEMOCAPDualStreamDataset`, `src/model.py:137-181`
`VADModelDualStream`) splits prior turns strictly by speaker identity relative to the
current utterance:
- `same_turns` — history from the same speaker as the current utterance ("self")
- `cross_turns` — history from the other speaker only ("cross")

Idea to test later: does the strict same/cross speaker stratification actually help,
or would one of the two streams do just as well (or better) if it saw the **full
dialogue history from both speakers** ("global") instead of being restricted to a
single speaker's turns?

Proposed variant to compare against current C3:
- Stream 1: self (same-speaker history only) — unchanged
- Stream 2: global — full prior dialogue history (both speakers combined,
  chronological), instead of cross-speaker-only

Compare against current self vs. cross split to see whether cross-speaker isolation
is pulling its weight, or whether just giving stream 2 the whole dialogue performs
comparably/better.

Not yet implemented — no code changes made. Revisit when ready to scope this as a
new context strategy / dataset variant.

## Archived pre-bugfix outputs

All prior `outputs/*` run directories (base, window, retrieval, dual_stream — both
roberta and xlmr) were produced before the pooler / optimizer / dual-stream-truncation
bug fixes (see repo history) and are moved to `outputs/archive_pre_bugfix/`. Their
metrics are not valid for the current codebase — re-run each config from scratch to
get numbers that reflect the fixed pooler representation, fixed Phase-2 fusion
optimizer, and bounded/left-truncated dual-stream context before drawing any
conclusions from them.
