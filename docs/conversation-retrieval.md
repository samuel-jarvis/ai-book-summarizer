# Conversation retrieval

`RetrievalService` returns the `k` nearest chunks (default 6) from the user's
project as the run's sources. It wraps `VectorStoreService.similarity_search`,
which checks project ownership, embeds the query, and searches only active,
completed generations that match the configured embedding model and dimensions.

```python
async with AsyncSessionLocal() as db:
    result = await RetrievalService(db, user_id).retrieve(
        rewritten_query, project_id=project_id, document_ids=[book_id]
    )

async with AsyncSessionLocal.begin() as db:
    service = ConversationService(db, user_id)
    run = await service.get_run(run_id)
    await service.checkpoint(
        run_id,
        stage="retrieved",
        rewritten_query=rewritten_query,
        snapshot=result.to_run_snapshot(run.snapshot),
    )
```

- `result.documents` holds full chunk text for the context builder.
- `result.sources` holds `SourceSnapshot`s labelled `S1`, `S2`, … in distance order,
  with 600-character excerpts.
- `document_ids` narrows the search. Ids outside the project match nothing; they
  do not raise.
- An unknown or foreign project raises `NotFoundError`. An empty result means
  nothing in scope is indexed yet.

## Not done yet

These can be added later, once evals show they're needed: deduplicating
overlapping chunks, a cosine-distance cutoff, guaranteed coverage per book for
comparisons, and per-document readiness reporting. The query embedding call
still runs inside the caller's DB session.
