# Chat eval experiment log

How the chat pipeline's quality is measured and how each change moved the numbers.
The benchmark itself is described in [chat/README.md](chat/README.md). Snapshots of
every recorded run live in [history/](history/).

## Results over time

Retrieval counts labeled quotes found among the retrieved chunks. Correct is the AI
judge's verdict. Rates are over the 12 cases (14 labeled quotes).

| Snapshot | Change | Comparison retrieval | Comparison correct | Overall retrieval | Overall correct |
|---|---|---|---|---|---|
| `baseline-v1` | none (starting point) | 2/8 | 0/3 | 7/14 | 8/12 |

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

### Baseline v1

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

### Experiment 1 (planned): reserve retrieval slots per book

- Hypothesis: comparison questions fail because one search over the whole project lets
  a single book fill all 6 slots. Guaranteeing each book its best 2 chunks, then filling
  the rest by overall rank, should raise comparison retrieval without changing the
  answer prompt.
- Change: `RetrievalService.retrieve` only. No extra model call.
- Success: comparison retrieval clearly above 2/8 and comparison correct above 0/3,
  with no drop in the direct and follow-up rows.
- Risk: a question about one book now carries 2 off-topic chunks.
- Result: not run yet.

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
