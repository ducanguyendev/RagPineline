"""
Inspect & Analyze Chunking Results

Usage:
    python inspect_chunks.py --input data/processed/chunked.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any
from collections import defaultdict
import statistics


def analyze_chunks(input_path: Path):
    """Analyze chunking output"""
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    # Collect stats
    chunks = []
    doc_stats = defaultdict(lambda: {"count": 0, "chars": 0, "tokens": 0})
    
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
                
                # Aggregate by doc
                doc_id = chunk.get("doc_id", "unknown")
                metrics = chunk.get("metrics", {})
                
                doc_stats[doc_id]["count"] += 1
                doc_stats[doc_id]["chars"] += metrics.get("char_count", 0)
                doc_stats[doc_id]["tokens"] += metrics.get("token_count", 0)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON decode error - {e}")
    
    if not chunks:
        print("❌ No chunks found")
        return
    
    # Extract metrics
    char_counts = [c.get("metrics", {}).get("char_count", 0) for c in chunks]
    token_counts = [c.get("metrics", {}).get("token_count", 0) for c in chunks]
    
    print("\n" + "="*70)
    print("📊 CHUNKING ANALYSIS")
    print("="*70)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total chunks:           {len(chunks):,}")
    print(f"  Total documents:        {len(doc_stats):,}")
    print(f"  Avg chunks per doc:     {len(chunks) / len(doc_stats):.1f}")
    
    print(f"\n📏 Character Count:")
    print(f"  Total:                  {sum(char_counts):,}")
    print(f"  Min:                    {min(char_counts):,}")
    print(f"  Max:                    {max(char_counts):,}")
    print(f"  Mean:                   {statistics.mean(char_counts):,.0f}")
    print(f"  Median:                 {statistics.median(char_counts):,.0f}")
    print(f"  Stdev:                  {statistics.stdev(char_counts) if len(char_counts) > 1 else 0:,.0f}")
    
    print(f"\n🔤 Token Count:")
    print(f"  Total:                  {sum(token_counts):,}")
    print(f"  Min:                    {min(token_counts):,}")
    print(f"  Max:                    {max(token_counts):,}")
    print(f"  Mean:                   {statistics.mean(token_counts):,.0f}")
    print(f"  Median:                 {statistics.median(token_counts):,.0f}")
    print(f"  Stdev:                  {statistics.stdev(token_counts) if len(token_counts) > 1 else 0:,.0f}")
    
    # LLM compatibility
    print(f"\n🤖 LLM Compatibility:")
    avg_tokens = statistics.mean(token_counts)
    print(f"  Avg tokens/chunk:       {avg_tokens:.0f}")
    
    models = {
        "GPT-3.5 (4K)": (4096, 1000),
        "GPT-4 (8K)": (8192, 1500),
        "Claude 3 (100K)": (100000, 10000),
        "Llama-2 (7B)": (4096, 500),
    }
    
    for model_name, (context, budget) in models.items():
        available = context - budget
        chunks_fit = available / avg_tokens
        fit_status = "✅" if chunks_fit >= 3 else "⚠️"
        print(f"    {fit_status} {model_name}: {chunks_fit:.1f} chunks fit")
    
    # Distribution analysis
    print(f"\n📊 Distribution:")
    percentiles = [10, 25, 50, 75, 90]
    sorted_tokens = sorted(token_counts)
    for p in percentiles:
        idx = int(len(sorted_tokens) * p / 100)
        val = sorted_tokens[idx]
        print(f"  {p}th percentile:        {val:,} tokens")
    
    # Quality metrics
    print(f"\n✅ Quality Checks:")
    tiny = sum(1 for c in char_counts if c < 100)
    huge = sum(1 for c in char_counts if c > 2000)
    print(f"  Chunks < 100 chars:     {tiny} ({tiny/len(chunks)*100:.1f}%)")
    print(f"  Chunks > 2000 chars:    {huge} ({huge/len(chunks)*100:.1f}%)")
    
    # Sample chunks
    print(f"\n🎯 Sample Chunks (first 3):")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  Chunk {i}:")
        print(f"    ID:                 {chunk.get('chunk_id')}")
        print(f"    Chars:              {chunk.get('metrics', {}).get('char_count', 'N/A'):,}")
        print(f"    Tokens:             {chunk.get('metrics', {}).get('token_count', 'N/A'):,}")
        
        content = chunk.get("chunk_content", "")
        preview = content[:200].replace("\n", " ")
        print(f"    Preview:            {preview}...")
    
    # Per-document summary
    print(f"\n📄 Per-Document Summary (top 5):")
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    for doc_id, stats in sorted_docs[:5]:
        avg_chars = stats["chars"] / stats["count"] if stats["count"] > 0 else 0
        avg_tokens = stats["tokens"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {doc_id}")
        print(f"    Chunks:             {stats['count']}")
        print(f"    Avg chars/chunk:    {avg_chars:,.0f}")
        print(f"    Avg tokens/chunk:   {avg_tokens:,.0f}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chunking results"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chunked.jsonl"),
        help="Chunked JSONL file"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample chunks to show"
    )
    
    args = parser.parse_args()
    
    analyze_chunks(args.input)


if __name__ == "__main__":
    main()
