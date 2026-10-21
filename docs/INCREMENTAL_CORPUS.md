# Incremental multi-document corpus

## Architecture

Phase 3 separates document processing from the canonical retrieval corpus:

```text
data/raw/<source.pdf>
  -> data/processed/documents/<document_id>/versions/<version>/
       pdf_extract.jsonl
       elements.jsonl
       image_captions.jsonl
       images/
       chunks.jsonl
  -> data/processed/corpus_manifest.json
  -> data/processed/chunked.jsonl
  -> data/vectorstore/chroma
```

`document_id` is deterministic and derived from the normalized relative source
path. The PDF SHA-256 is separate: it detects content changes while the document
identity remains stable.

The processing fingerprint includes the PDF SHA-256, chunk strategy, target
tokens, overlap, and the pipeline schema version. Changing either the file or
chunk configuration makes the document `changed`.

## Manifest and per-document artifacts

`corpus_manifest.json` is the ownership registry for the corpus. Every indexed
document records its source path, PDF hash, file size, processing fingerprint,
embedding model, chunk count, exact chunk IDs, active per-document `chunks.jsonl`,
status, and timestamps.

Artifacts are immutable versions. A new or changed document is processed in a
staging directory and promoted to a new version only after validation and vector
indexing. Older versions remain available for recovery; explicit cleanup is a
later maintenance task.

## Canonical corpus and BM25

`data/processed/chunked.jsonl` is the atomic aggregate of every active document
in the manifest. It is rebuilt from per-document chunk files after a successful
update. BM25 continues to read this file, and its cache is cleared after every
canonical replacement. Dense Chroma and BM25 are therefore validated against
the same active chunk set.

Before canonical replacement the manager rejects missing IDs, per-document or
global duplicate IDs, missing artifact files, manifest count mismatches, and
Chroma/manifest count mismatches.

## Incremental flows

- **New:** process only the selected PDF, upsert its chunks, append it to the
  manifest-backed canonical corpus, and leave all existing documents untouched.
- **Unchanged:** when the fingerprint matches, skip Docling, BLIP, chunking,
  embedding, Chroma mutation, and file rewrites.
- **Changed:** process a new artifact version, upsert current IDs, then delete
  only `old_ids - new_ids`. Other documents are never deleted.

If a vector or file commit fails, the manifest is not marked successful. After a
vector mutation, the manager restores the previous document vectors and the old
canonical bytes when possible.

Incremental indexing is refused when the manifest embedding model differs from
the configured `EMBEDDING_MODEL`; mixing embedding spaces requires a full corpus
rebuild.

## Bootstrap the existing corpus

The current production corpus has 5 documents and 3627 stable-ID vectors.
Bootstrap must not rebuild or re-embed them. The completed dry run verified
3627 canonical chunk IDs, 3627 unique IDs, 5 detected documents, and 3627
matching Chroma vectors. The metadata-only `--apply` has been run successfully.

First run the read-only inspection:

```powershell
.\.venv-xpu\Scripts\python.exe .\scripts\bootstrap_corpus_manifest.py --dry-run
```

For a future migration, review the dry-run counts and matching ID check, stop
the backend so no indexing request can run concurrently, then create only the
manifest and per-document chunk files:

```powershell
.\.venv-xpu\Scripts\python.exe .\scripts\bootstrap_corpus_manifest.py --apply
```

Bootstrap never initializes the embedding model or calls Chroma mutation APIs.
It leaves the existing canonical JSONL byte-for-byte unchanged and checks the
read-only Chroma IDs before and after.

## Full rebuild semantics

Full rebuild is a separate explicit action. It removes every current vector and
re-embeds the complete canonical corpus represented by the manifest. It never
rebuilds only the PDF selected in the UI. The checkbox defaults to off and
requires confirmation. If an error occurs, the manager attempts to restore the
full canonical vector set before reporting the failure.

Back up `data/vectorstore/chroma`, `data/processed/corpus_manifest.json`, and
`data/processed/documents` before a planned full rebuild or manual recovery.

## Manually add one PDF after bootstrap

1. Copy one new PDF into `data/raw`.
2. Start the backend and frontend in separate PowerShell windows:

   ```powershell
   .\.venv-xpu\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 5000 --reload
   ```

   ```powershell
   cd frontend
   npm run dev -- --host 127.0.0.1
   ```
3. Select the PDF. Its UI status should be `NEW`.
4. Leave **Full rebuild vector database** off.
5. Click **Index Selected Document**.
6. Verify vectors increased by only that document's chunk count.
7. Run the same PDF again and verify status `UNCHANGED` and vector count unchanged.
8. Query content from an old PDF and the new PDF to confirm dense and BM25 access
   to the complete corpus.

Explicit document deletion and old artifact-version garbage collection are
intentionally deferred.
