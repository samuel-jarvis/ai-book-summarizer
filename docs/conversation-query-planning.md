# Query planning (splitting multi-part questions)

A single search embeds the whole question as one vector. A question that asks about
two things, such as "how did X end in book A and Y in book B", lands between them
and matches neither well (see evals/EXPERIMENTS.md, experiment 1). `QueryPlanService`
splits such a question into focused searches before retrieval.

## Flow

1. The rewrite stage still produces one standalone query and the stored message is
   untouched.
2. `QueryPlanService.plan(query)` makes one short, temperature-zero model call that
   returns `{"queries": [...]}`: up to `RAG_MAX_SUBQUERIES` (default 3) queries, one
   per distinct topic, character or book, each standing alone.
3. A single returned query means a single topic. The original text is kept, not the
   model's paraphrase, so single-topic questions behave as before.
4. Two or more queries go to `RetrievalService.retrieve(..., queries=[...])`, which
   calls `VectorStoreService.multi_query_search`. Each query is embedded and searched
   separately; `merge_groups` gives each an equal share of the `k` chunks (shared
   chunks free a slot, leftover slots go to the nearest remaining candidates).
5. The queries that ran are saved in `RunSnapshot.search_queries`.

The per-book minimum (`RAG_MIN_CHUNKS_PER_DOCUMENT`) applies only to the single-query
path, because each focused query already targets its own topic.

## Failure handling

Malformed JSON, empty or multi-line queries, or any provider error fall back to the
single original query. Provider text is never stored. Cancellation propagates.
`RAG_MAX_SUBQUERIES=1` skips the call entirely.

## Cost

One extra short model call per turn, before retrieval, on every question including
single-topic ones. Measure it before keeping this on by default.
