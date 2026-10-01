# Chroma index lifecycle

## Safe default

Embedding now uses stable Chroma document IDs from each chunk's `chunk_id` and
upserts through the public LangChain Chroma API. Re-indexing the same chunks
therefore updates those IDs instead of appending duplicate vectors.

The backend and frontend default to `reset_db: false`. The UI only deletes the
current vector database when **Full rebuild vector database** is explicitly
enabled and the confirmation warning is accepted.

If a legacy chunk has no `chunk_id`, the indexer logs a warning and derives a
deterministic fallback ID from stable document/position metadata and a content
hash. Duplicate IDs in one indexing input are rejected because IDs must be
globally unique within the `rag_documents` collection.

## Existing production database

The production database has completed the stable-ID rebuild and currently has
5006 vectors for 5006 unique chunk IDs. Phase 3 bootstraps document ownership
metadata around those vectors without re-embedding them. See
`docs/INCREMENTAL_CORPUS.md` for the manifest and incremental workflow.

## Adding or updating chunks after migration

With a stable-ID database, leave **Full rebuild vector database** off for normal
indexing. Existing `chunk_id` values are updated and new `chunk_id` values are
inserted. A full rebuild remains an explicit recovery or planned migration
operation, not the normal indexing path.

## Safe idempotency check

Run:

```powershell
.\.venv-xpu\Scripts\python.exe .\scripts\test_index_idempotency.py
```

The test samples 20 chunks, uses deterministic test embeddings, and creates a
temporary Chroma directory separate from `data/vectorstore/chroma`. It indexes
the same sample twice and requires the vector count to remain unchanged.
