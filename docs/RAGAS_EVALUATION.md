# RAGAS evaluation

Activate `.venv-xpu`, set `GEMINI_API_KEY` in the project `.env`, and confirm the indexed corpus covers the evaluation questions. Run a small sample first:

```powershell
python -m src.eval.ragas_evaluator --limit 3
```

Results are written to `data/ragas_results.json`. Use `--include-contexts` to include retrieved text in the output, or `--dataset` and `--output` to select paths.
