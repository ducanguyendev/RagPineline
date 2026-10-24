# RAGAS evaluation

The active evaluation dataset is `data/eval_dataset.json`. It is generated from the reviewed `Luật LĐ` worksheet in `RAG_TestCase (1).xlsx` with the importer below. Gold `Evidence` values from Excel are stored in `reference_contexts`; they remain separate from the contexts retrieved by RagPipeline.

Import or refresh the dataset with:

```powershell
python -m src.eval.import_excel_eval_dataset `
  --input "RAG_TestCase (1).xlsx" `
  --sheet "Luật LĐ" `
  --output data/eval_dataset.json
```

Activate `.venv-xpu` and set `GEMINI_API_KEY` in the project `.env`. Run small samples first:

```powershell
python -m src.eval.ragas_evaluator --limit 1
python -m src.eval.ragas_evaluator --limit 3
```

The full evaluation is available with `python -m src.eval.ragas_evaluator`; run it intentionally because each case invokes the RAG pipeline and Gemini/RAGAS judge calls repeatedly. Results are written to `data/ragas_results.json`. Use `--include-contexts` to include retrieved text in the output, or `--dataset` and `--output` to select paths.
