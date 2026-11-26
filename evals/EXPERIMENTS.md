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
| `exp2-split-queries` | 23 | question split into focused searches | 4/14 (007 failed) | 0/6 (+1 failed) | 12/25 | 13/22 |
| `exp2-split-queries-holdout` | 6 | question split into focused searches | 2/2 (023 failed) | 1/1 (+1 failed) | 5/5 | 5/5 |
| `exp3-c300-k6` | 23 | 300-token chunks, top 6 | 3/16 | 0/7 | 9/27 | 11/23 |
| `exp3-c300-k6-holdout` | 6 | 300-token chunks, top 6 | 2/2 (023 failed) | 1/1 (+1 failed) | 4/5 | 4/5 |
| `exp3-c300-k16` | 23 | 300-token chunks, top 16 | 3/10 (3 failed) | 0/4 (+3 failed) | 11/21 | 12/20 |
| `exp3-c300-k16-holdout` | 6 | 300-token chunks, top 16 | 2/4 | 1/2 | 4/7 | 4/6 |

Retrieval and correct counts leave out cases whose answer failed, so a failure can make
a run look better or worse than it is. Like-for-like retrieval over the same 20
comparison quotes, recomputed from each report's retrieved chunks including the failed
cases: `baseline-v2` 6/20, `exp1` 7/20, `exp2` 8/20.
Over all 34 labeled quotes (dev plus holdout, failed cases included): `baseline-v2` 17/34,
`exp3-c300-k6` 13/34, `exp3-c300-k16` 17/34.

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

### Experiment 1: reserve retrieval slots per book (dropped)

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

### Experiment 2: split the question into one search per part (dropped)

- Date: 2026-11-26. Snapshots: `exp2-split-queries`, `exp2-split-queries-holdout`.
- Hypothesis: experiment 1 left the other book's reserved chunks wrong because one
  blended two-topic query matches neither topic. A focused query per part should rank
  each part's passage higher.
- Change: `QueryPlanService` asks the model for up to 3 standalone queries; retrieval
  searches each and `merge_groups` gives each an equal share of the 6 chunks. A
  single-topic question keeps its original text. Setting `RAG_MAX_SUBQUERIES` (1 = off).
  Commit `d744471`, clean tree. 14 new tests; full suite 203 passed against Postgres.
- Mechanism: the plans are good and readable. For 007 the model produced "What did
  onlookers see when the Time Traveller demonstrated his small model in The Time
  Machine?" and "What did onlookers see when Griffin removed his disguise in The
  Invisible Man?". Every cross-book comparison got a 3 + 3 split. The one follow-up
  comparison (holdout 024) was left as a single query.
- Result: a small gain, not a fix. Like-for-like comparison quotes found over all 20
  anchors, failed cases included: baseline-v2 6/20, exp1 7/20, exp2 8/20. Comparison
  correct is 0/6 on dev (one more failed) and 1/1 on holdout (one failed); the
  holdout's apparent 100% retrieval is two of its four quotes, because the other case
  failed. Do not read it as a win.
- Why it is only a small gain: with focused queries the evidence is usually in the
  right book but not near the top. Within-book rank of each of the 20 comparison
  quotes under the sub-queries (measured offline over the top 60 results per query;
  ranks are approximate and a "miss" can be a cut-off artifact):
  top 3: 8/20, top 6: 9/20, top 10: 12/20, top 20: 16/20, top 60: 17/20.
  Examples: 008 gets *Time Machine* p51 at rank 13 and *Invisible Man* p29 at 17;
  026 gets 15 and 13; 009's p86 ending at 8. Giving each part 3 chunks cannot reach
  rank 13. The limit is how well embedding similarity ranks an exact detail inside an
  800-token chunk, not coverage or query wording.
- New cost and risk: one extra model call per turn, before retrieval. And two of nine
  comparison answers ended in `answer_truncated` (007 and holdout 023), against one in
  `baseline-v2` and none in exp1. The 2000-token output cap is hit on long two-part
  comparison answers, and when that happens the turn fails and its retrieval score is
  lost. This is now the most user-visible defect the eval has found.
- Decision: do not merge to main. The split is sound but costs a call per turn for +2
  of 20 quotes. Keep it on the branch as a component for the next experiment.
- Next, in order of expected value from the rank data: (1) smaller chunks (for example
  300 tokens; bump `rag_chunking_version`, re-index) because the evidence is rank 8 to
  17 inside big chunks; (2) retrieve about 15 per part and rerank down to 3; (3) hybrid
  keyword plus vector search for rare names such as Marvel or Iping. Separately, fix
  the truncation (raise or adapt the output cap, or ask for a shorter answer), and make
  the eval score retrieval even when the answer fails.
- What this does not prove: that splitting is useless once ranking improves. It may
  pay off combined with a better ranker, and that is how it should be tested.

### Simulation: three-turn chat (book A, book B, then both)

- Date: 2026-11-26. Question: in a realistic chat, where the user asks about each book
  first, do experiments 1 and 2 help on the final cross-book question?
- Setup: two scenarios, each three turns in one conversation. Scenario 1: who was Weena
  and how did she meet the Time Traveller; who was Marvel and how did he get involved
  with Griffin; how did Weena warn the Time Traveller about the dangers of the night and
  what practical help did Griffin insist Marvel give him. Scenario 2: what did the Time
  Traveller find in the Palace of Green Porcelain; who was Marvel and why did Griffin need
  him; what condition were the old books in at the palace and what did Marvel carry when
  he fled Iping. The final questions reuse the labels of `chat-026` and `chat-008`.
  Each ran twice under three settings: off (main's behaviour), per-book slots, and
  split queries.
- Result: all 12 runs found 0 of the labeled quotes and none was judged correct. Every
  answer honestly said the retrieved passages did not contain the detail. The chunk
  split on the final question was 4/2, 5/1 or 6/0 with the settings off (one run
  starved a book completely), 3/3 or 4/2 with slots, and 3/3 every time with splitting,
  so the mechanisms worked and made no difference to the quotes found.
- Observation: the history-aware rewrite already adds the book titles ("In The Time
  Machine, ... in The Invisible Man, ..."), so the chat flow gets focusing for free.
  One split run reached page 44 for a fact on page 43, a chunk-boundary near miss.
- Caveats: the labels come from the two cases that failed in every run, so this
  scenario could not separate the settings; 2 scenarios and 2 trials is indicative only.
- Decision (2026-11-26): drop experiments 1 and 2. Per-book slots do not scale past two
  books (a minimum of 2 per book fills all 6 slots with three books). Splitting costs a
  model call per turn for about two extra quotes of 20. The code stays on the pushed
  branch `feat/per-book-retrieval` (commits `652f446` and `d744471`) because the
  snapshots reference them. Next: smaller chunks (experiment 3), because the missing
  evidence ranks 8 to 17 inside 800-token chunks and that fix helps with any number
  of documents.

### Experiment 3: smaller chunks (dropped)

- Date: 2026-11-26. Snapshots: `exp3-c300-k6`, `exp3-c300-k16` and their `-holdout` pairs.
  Code: commit `842fc4e` on the pushed branch `feat/smaller-chunks` (it only makes the
  retrieval count a setting, `RAG_TOP_K`, default 6, and records chunking and top-k in
  run stamps). Chunk size was changed through environment variables.
- Hypothesis: the missing evidence ranks 8 to 17 because exact sentences are buried in
  800-token chunks, so 300-token chunks should rank them higher.
- End-to-end result: worse, or no better at the same context size. Like-for-like quotes
  found over all 34 (failed cases included): baseline-v2 (800 tokens, top 6) 17/34;
  300 tokens with top 6, 13/34; 300 tokens with top 16 (about the same context tokens),
  17/34. Overall retrieval counts 9/27 and 11/21 against 12/25. With top 16, three
  comparison answers ended in `answer_truncated`.
- Offline check on fixed queries, to remove generation noise: for each of the 34 quotes
  I ranked its own book's chunks using the baseline run's final query (rewrite included),
  on four indexes of the same two books. Quotes found within rank N (of 34):

  | Chunk tokens (overlap) | top 3 | top 6 | top 10 | top 16 |
  |---|---|---|---|---|
  | 300 (50) | 9 | 16 | 18 | 20 |
  | 500 (80) | 15 | 18 | 21 | 27 |
  | 800 (120), current | 19 | 21 | 26 | 29 |
  | 1200 (150) | 20 | 22 | 25 | 29 |

  Smaller is clearly worse; 800 to 1200 sit on a plateau. At about equal context
  (800 x 6, 500 x 10, 300 x 16) they find 21, 21 and 20, so there is no gain from
  re-chunking. The hypothesis is refuted: chunk size is not the lever.
- What the check does show: the current setup finds 21 of 34 quotes in its book's top
  6 but 29 of 34 in the top 16, so a good reranker over about 16 candidates has up to
  8 quotes (24 points) of headroom. It also says what picking the book would be worth:
  ranking inside the correct book finds 21/34 at top 6 against 17/34 end to end.
- Decision: keep 800/120; do not merge a chunk-size change. Next experiment:
  retrieve about 16 and rerank to 6.
- Caveats: two books and one chunking strategy (token splitter); a quote counts only if
  it sits wholly in one chunk, which can penalise small chunks slightly (the overlap of
  50 tokens exceeds a typical quote); the offline ranks assume the right book is known.
  Extra indexes for 300, 500 and 1200 tokens remain in the eval database.

## Backlog

Ideas to build later, not yet scheduled.

- Let the user choose which book(s) a question is answered from. Retrieval already
  accepts `document_ids`; this needs a field on the message request, a picker in the UI,
  and eval cases that scope a question to one book. Exact, free, and independent of how
  many books a project holds.
- When the answer says it could not find the detail, offer a recheck: "Search in a
  specific book" with a picker, then rerun the turn scoped to the chosen document(s)
  (the existing run retry is the natural hook). It needs a reliable "not found" signal,
  better as a flag from the answer stage than parsed from prose; the eval can measure
  how often it fires and how often the scoped rerun finds the answer.
- Fix `answer_truncated`: two of nine comparison answers in experiment 2 hit the
  2000-token output cap and the turn failed. Raise or adapt the cap, or ask for shorter
  answers.
- Score retrieval even when generation fails, so a failed turn no longer drops out of
  the retrieval metrics.
- Retrieve about 16 candidates and rerank to 6 (experiment 4; headroom up to 24 points
  on the current 34 quotes, see experiment 3); hybrid keyword plus vector search for rare
  names such as Marvel and Iping.

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
