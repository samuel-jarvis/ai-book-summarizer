# Chat eval experiment log

How the chat pipeline's quality is measured and how each change moved the numbers.
The benchmark itself is described in [chat/README.md](chat/README.md). Snapshots of
every recorded run live in [history/](history/).

## Results over time

Retrieval counts labeled quotes found among the retrieved chunks. Correct is the AI
judge's verdict over completed cases. Snapshots are only comparable when they use the
same case labels (the `cases_hash` in each snapshot).

| Snapshot | Cases | Change | Comparison retrieval | Comparison correct | Overall retrieval | Overall correct |
|---|---|---|---|---|---|---|
| `baseline-v1` (superseded) | 12 | none (starting point) | 2/8 | 0/3 | 7/14 | 8/12 |
| `baseline-v2` | 23 | none; more cases | 4/14 | 1/6 (+1 failed) | 12/25 | 14/22 |
| `baseline-v2-holdout` | 6 | none; held-out cases | 2/4 | 1/2 | 5/7 | 5/6 |
| `exp1-per-book-slots` | 23 | per-book slots (min 2 per book) | 5/16 | 0/7 | 13/27 | 13/23 |
| `exp1-per-book-slots-holdout` | 6 | per-book slots (min 2 per book) | 2/4 | 0/2 | 5/7 | 4/6 |

## Method

1. Branch from `main`, and start from a clean tree so the run is tied to a commit.
2. Run `uv run python -m evals.chat`. Every report records the git commit, whether
   the tree was dirty, a hash of the case labels and the judge model.
3. Save a snapshot: `uv run python -m evals.chat.summary --save NAME --note "what changed"`.
   Full reports stay local (they contain book text); snapshots are committed.
4. Compare: `uv run python -m evals.chat.compare evals/history/baseline-v1.json evals/history/NAME.json`.
   It lists metric changes by case type and every case whose scores flipped, and warns
   if the labels changed or the run came from a dirty tree.
5. Add an entry below, whether the change worked or not, and commit the snapshot with
   the code change. Put the numbers in the commit message.

Rules: change one thing per experiment, do not edit case labels after seeing results
(if a label is wrong, say so in the entry and start a new baseline), and check the
direct and follow-up rows for regressions on every change.

## Entries

### Baseline v1 (superseded by v2)

- Date: 2026-11-25. Snapshot: `baseline-v1`.
- Code: `main` at 248c06f plus the then-uncommitted eval harness.
- Result: direct 3/3 correct, follow-up 2/3, comparison 0/3, unanswerable 3/3 (declined
  without inventing). Retrieval 7/14 overall; comparison 2/8.
- Diagnosis from the report's retrieved chunks, per comparison case:
  - chat-007: all 6 chunks came from *The Time Machine*; none from *The Invisible Man*.
  - chat-008: 1 and 5. The five from *The Invisible Man* were on the wrong pages, so
    the combined two-topic question also blurred the search.
  - chat-009: 5 and 1. The single *Invisible Man* chunk was page 77; the ending is page 86.
- Caveats: only 3 comparison cases. The judge is the same model that writes the answers.
  This report predates run stamping, so its snapshot carries a note instead of a commit.

### Baseline v2 (supersedes v1)

- Date: 2026-11-25. Snapshots: `baseline-v2` (23 cases) and `baseline-v2-holdout` (6 cases).
- Why: v1 had only 3 comparison cases. The labels changed (23 cases = the original 12
  plus 11 new; 6 more held out), so v1 is no longer comparable. One v1 label was
  corrected before any run: the `chat-020` quote had a typo (`oﬀmy`) and was shortened.
- Code: `main` at 8687c31. The new case files were still uncommitted when it ran, so the
  snapshots carry `git_dirty: true` for that reason only.
- Result (dev): direct 5/6 correct, follow-up 3/5, comparison 1/6 plus one failed case,
  unanswerable 5/5. Retrieval 12/25. Holdout: 5/6 correct, retrieval 5/7.
- The starved-book pattern from v1 repeats. Of the 8 cross-book comparison cases that
  completed, 5 had one book at 0 or 1 of the 6 retrieved chunks (007: 6/0, 008: 1/5,
  009: 5/1, 021: 6/0, holdout 023: 6/0). The *Invisible Man* side was missed in every
  one except holdout 024.
- Not a per-book problem, so per-book slots are not expected to fix these: `chat-015`
  and `chat-020` (single-book detail missed although the right book supplied all or
  most chunks), `chat-006` (follow-up), and `chat-025` (same-book control, 6 chunks
  from one book, one of two quotes found).
- A separate bug the eval exposed: `chat-026` failed with `answer_truncated`. The model
  hit the 2000 output-token cap (`finish_reason: length`) on a long comparison answer,
  and the pipeline publishes only on a clean stop. This is a generation limit, not
  retrieval; track it separately.
- Caveats: same-model judge; 23 + 6 cases is still small, so a one-case swing is about
  4 points (17 on holdout).

### Experiment 1: reserve retrieval slots per book (not merged to main)

- Hypothesis: comparison questions fail because one search over the whole project lets
  a single book fill all 6 slots. Guaranteeing each book its best 2 chunks, then filling
  the rest by overall rank, should raise comparison retrieval without changing the
  answer prompt.
- Change: `RetrievalService.retrieve` only. No extra model call.
- Success, measured against `baseline-v2`: comparison retrieval clearly above 4/14 and
  comparison correct above 1/6, with no drop in the direct and follow-up rows, then
  confirmed on `baseline-v2-holdout` (comparison retrieval above 2/4).
- Risk: a question about one book now carries 2 off-topic chunks.
- Implementation: `VectorStoreService.balanced_search` and the pure `merge_balanced`,
  controlled by `RAG_MIN_CHUNKS_PER_DOCUMENT` (default 2; 0 restores plain nearest-k).
  Commit `652f446`, run from a clean tree. 9 new tests; the retrieval test file passes
  against Postgres (16 tests).
- Result: the mechanism worked, the outcome did not. Success criteria were not met.
  - Distribution is fixed: every cross-book case now has at least 2 chunks from each
    book (007: 6/0 became 4/2, 021: 6/0 became 4/2, 009: 5/1 became 4/2; holdout 023:
    6/0 became 4/2).
  - Retrieval barely moved: comparison 4/14 to 5/16 (one more quote found, `chat-021`);
    holdout 2/4 unchanged. Comparison correct went 1/6 to 0/7 on dev and 1/2 to 0/2 on
    holdout. The dev denominators differ because `chat-026` completed this time.
  - Why: the reserved chunks from the other book are the wrong ones. For the combined
    two-topic question the other book's best 2 chunks are not the evidence: 007 got
    pages 73 and 77 of *The Invisible Man* but the evidence is page 20; 009 got 50 and
    77 but the ending is page 86; 026 got the right book but not pages 27 and 43.
    Coverage was necessary but not sufficient: the blurred combined query is the
    remaining problem.
  - Single-book cost: `chat-025` (same-book control) now carries 2 off-topic
    *Time Machine* chunks in place of 2 *Invisible Man* chunks; its retrieval was
    unchanged (1/2) but its answer flipped to wrong. One sample, so this is a risk
    signal, not proof.
- Noise to keep in mind: the query rewrite is not deterministic even at temperature 0.
  Rewrites differed between runs for `chat-005`, `-006`, `-020` and `-024`, so flips in
  cases with prior turns are not evidence for or against the change. Example: holdout
  `chat-024` went from 2/2 to 1/2 retrieval, but its books already met the minimum, so
  the merge changed nothing; only the rewrite wording changed. Cases without prior
  turns have deterministic retrieval, so `chat-021`'s +1 is real. Citation and
  support scores also moved on some cases between runs with no retrieval change.
- Decision: do not merge to main as an improvement. Keep the branch as the base for
  Experiment 2 (decomposing the question into one focused search per part, each with
  the per-book minimum), which targets the actual remaining cause.
- What this does not prove: that a larger minimum (3 or 4 per book) would not help;
  that is a cheap follow-up (`RAG_MIN_CHUNKS_PER_DOCUMENT=3`) but treat it as tuning,
  and confirm any gain on the holdout.

## Template

```
### Experiment N: short name
- Date / snapshot / commit:
- Hypothesis:
- Change:
- Before -> after (paste the compare output for the rows that moved):
- Cases that flipped, either way:
- Decision (kept / reverted) and why:
- What this does not prove:
```
