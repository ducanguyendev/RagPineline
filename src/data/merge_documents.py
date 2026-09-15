"""
Merge text documents and image captions for RAG.

Input:
    pdf_extract.jsonl (text from PDF)
    image_captions.jsonl (image captions)
    image_metadata.jsonl (image locations)

Output:
    merged_documents.jsonl
    
Process:
    1. Load all text documents
    2. Load all image captions  
    3. Link images to pages
    4. Merge: for each page, include text + images from that page
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(
    file_path: Path
) -> list:
    """Load JSONL file into list"""
    
    records = []
    
    if not file_path.exists():
        print(
            f"⚠ File not found: {file_path}"
        )
        return records
    
    with file_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                records.append(
                    json.loads(line)
                )
    
    return records


# ============================================================
# ORGANIZE BY PAGE
# ============================================================

def organize_by_page(
    text_records: list,
    caption_records: list,
    metadata_records: list
) -> dict:
    """
    Organize documents by page number
    
    Returns:
        {page: {text: ..., images: [...]}}
    """
    
    pages = defaultdict(
        lambda: {
            "text": "",
            "images": []
        }
    )
    
    # Add text content
    for record in text_records:
        page = record.get(
            "metadata", {}
        ).get("page", 0)
        
        if page > 0:
            pages[page]["text"] = record.get(
                "content",
                ""
            )
    
    # Group captions by page
    captions_by_page = defaultdict(list)
    
    for caption in caption_records:
        page = caption.get("page", 0)
        captions_by_page[page].append(caption)
    
    # Add images to pages
    for page, captions in captions_by_page.items():
        if page > 0:
            pages[page]["images"] = captions
    
    return pages


# ============================================================
# MERGE DOCUMENTS
# ============================================================

def merge_documents(
    text_path: Path,
    caption_path: Path,
    metadata_path: Path,
    output_path: Path,
    include_images: bool = True
):
    """
    Merge text documents with image captions
    
    Args:
        text_path: pdf_extract.jsonl
        caption_path: image_captions.jsonl
        metadata_path: image_metadata.jsonl
        output_path: merged_documents.jsonl
        include_images: Whether to include images in output
    """
    
    print("Loading documents...")
    
    text_records = load_jsonl(text_path)
    caption_records = load_jsonl(caption_path)
    metadata_records = load_jsonl(metadata_path)
    
    print(
        f"✓ Text records: {len(text_records)}"
    )
    print(
        f"✓ Captions: {len(caption_records)}"
    )
    print(
        f"✓ Image metadata: {len(metadata_records)}"
    )
    
    # Organize by page
    print("\nOrganizing by page...")
    
    pages = organize_by_page(
        text_records,
        caption_records,
        metadata_records
    )
    
    print(f"✓ {len(pages)} pages with content")
    
    # Create merged documents
    print("\nMerging documents...")
    
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    merged_records = []
    image_count = 0
    
    for page_no in sorted(pages.keys()):
        
        page_data = pages[page_no]
        text_content = page_data["text"]
        images = page_data["images"]
        
        # Skip pages with no text
        if not text_content.strip():
            continue
        
        # Base record
        record = {
            
            "id":
                f"page_{page_no:03d}",
            
            "page":
                page_no,
            
            "content":
                text_content,
            
            "type":
                "text+images" if images else "text"
        
        }
        
        # Add images if requested
        if include_images and images:
            
            record["images"] = [
                {
                    "image_id":
                        img["image_id"],
                    
                    "caption":
                        img["caption"],
                    
                    "image_path":
                        img["image_path"]
                }
                for img in images
            ]
            
            image_count += len(images)
        
        merged_records.append(record)
    
    # Write output
    print(f"\nWriting merged documents: {output_path}")
    
    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        
        for record in merged_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=None
                )
                + "\n"
            )
    
    print("\n" + "=" * 60)
    print("DONE")
    print(
        f"Merged documents: {len(merged_records)}"
    )
    
    if include_images:
        print(
            f"Images included: {image_count}"
        )
    
    print(
        f"Output: {output_path}"
    )
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    
    parser = argparse.ArgumentParser(
        description="Merge text and images for RAG"
    )
    
    parser.add_argument(
        "--text",
        type=Path,
        default=Path(
            "data/processed/pdf_extract.jsonl"
        ),
        help="Text extract from PDF"
    )
    
    parser.add_argument(
        "--captions",
        type=Path,
        default=Path(
            "data/processed/image_captions.jsonl"
        ),
        help="Image captions"
    )
    
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/image_metadata.jsonl"
        ),
        help="Image metadata"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/merged_documents.jsonl"
        ),
        help="Merged output"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't include images in output"
    )
    
    args = parser.parse_args()
    
    merge_documents(
        args.text,
        args.captions,
        args.metadata,
        args.output,
        include_images=not args.no_images
    )


if __name__ == "__main__":
    main()
