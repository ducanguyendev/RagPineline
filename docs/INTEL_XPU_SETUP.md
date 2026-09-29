# Intel XPU setup for index building

This setup accelerates only the BGE-M3 index build. Retrieval, reranking,
chunking, and application data formats are unchanged.

## 1. Create and activate a Python 3.12 environment

```powershell
py -3.12 -m venv .venv-xpu
.\.venv-xpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install PyTorch with Intel XPU support

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Do not replace this XPU wheel with a CPU-only PyTorch build.

## 3. Install project dependencies

`requirements.txt` intentionally does not pin or reinstall PyTorch.

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

## 4. Check the accelerator

```powershell
python scripts/check_accelerator.py
```

On an Intel Arc A370M, the selected device should be `xpu:0`.

## 5. Benchmark CPU and XPU

Run a small CPU baseline:

```powershell
python scripts/benchmark_embedding.py --device cpu --batch-size 8 --limit 20
```

Test conservative XPU batches before increasing them on a 4 GB GPU:

```powershell
python scripts/benchmark_embedding.py --device xpu:0 --batch-size 4 --limit 20
python scripts/benchmark_embedding.py --device xpu:0 --batch-size 8 --limit 20
python scripts/benchmark_embedding.py --device xpu:0 --batch-size 16 --limit 20
```

Use the largest stable batch with the best measured throughput. Batch 8 is
the safe default for a 4 GB Arc A370M.

## 6. Build the index with Intel Arc

Set the indexing configuration in the current PowerShell session:

```powershell
$env:INDEX_DEVICE = "xpu:0"
$env:INDEX_DTYPE = "float16"
$env:INDEX_BATCH_SIZE = "8"
$env:CHROMA_BATCH_SIZE = "128"
```

Then run the index build:

```powershell
python .\src\rag\indexer.py --input .\data\processed\chunked.jsonl --db .\data\vectorstore\chroma
```

If the existing Chroma directory must be rebuilt, stop the backend before
removing it so Windows does not keep Chroma files locked.

## Configuration reference

- `INDEX_DEVICE=auto`: prefer XPU, then CUDA, then CPU.
- `INDEX_DTYPE=auto`: use float16 on XPU/CUDA and float32 on CPU.
- `INDEX_BATCH_SIZE=8`: SentenceTransformer encoding batch size.
- `CHROMA_BATCH_SIZE=128`: number of documents inserted into Chroma per call.

The embedding batch and Chroma insertion batch are independent settings.
