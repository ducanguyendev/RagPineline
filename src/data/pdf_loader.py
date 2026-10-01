"""
PDF Loader for RAG

Parse PDF using Docling and export structured elements:
- data/processed/elements.jsonl (Source of truth)
- data/images/<document_id>/fig_XXX.png (Extracted physical images)
- data/processed/pdf_extract.jsonl (Backward compatibility)
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_elements_from_docling(
    doc,
    pdf_stem: str,
    images_dir: Path,
    image_path_prefix: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Duyệt toàn bộ các phần tử (elements) do Docling bóc tách,
    trích xuất và lưu các file ảnh PNG thực tế vào data/images/<document_id>/.
    """
    elements = []
    image_records = []
    elem_count = 0
    fig_count = 0

    images_dir.mkdir(parents=True, exist_ok=True)

    for item, _level in doc.iterate_items():
        elem_count += 1
        item_class = item.__class__.__name__

        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no

        item_text = ""
        table_data = None
        image_info = None
        elem_type = "paragraph"

        item_lower = item_class.lower()

        if "section" in item_lower or "header" in item_lower or "title" in item_lower:
            elem_type = "heading"
            if hasattr(item, "text"):
                item_text = item.text
        elif "table" in item_lower:
            elem_type = "table"
            try:
                item_text = item.export_to_markdown(doc)
            except Exception:
                item_text = getattr(item, "text", "")
            table_data = {"markdown": clean_text(item_text)}
        elif "picture" in item_lower or "image" in item_lower or "figure" in item_lower:
            elem_type = "image"
            fig_count += 1
            item_text = getattr(item, "caption", "") or getattr(item, "text", "")
            
            image_id = f"{pdf_stem}_fig_{fig_count:03d}"
            filename = f"fig_{fig_count:03d}.png"
            abs_image_path = images_dir / filename
            relative_prefix = image_path_prefix or f"data/images/{pdf_stem}"
            rel_image_path = f"{relative_prefix.rstrip('/')}/{filename}"

            width, height = 0, 0
            saved = False

            if hasattr(item, "get_image"):
                try:
                    img = item.get_image(doc)
                    width, height = img.size
                    img.save(abs_image_path)
                    saved = True
                except Exception as exc:
                    print(f"! Failed to extract image {image_id}: {exc}")

            image_info = {
                "image_id": image_id,
                "image_path": rel_image_path if saved else "",
                "width": width,
                "height": height,
                "caption": clean_text(item_text)
            }

            if saved:
                image_records.append({
                    "document_id": pdf_stem,
                    "image_id": image_id,
                    "image_path": rel_image_path,
                    "page": page_no,
                    "width": width,
                    "height": height,
                    "caption": clean_text(item_text)
                })

        elif "caption" in item_lower:
            elem_type = "caption"
            item_text = getattr(item, "text", "")
        else:
            elem_type = "paragraph"
            if hasattr(item, "text"):
                item_text = item.text
            elif hasattr(item, "export_to_markdown"):
                try:
                    item_text = item.export_to_markdown()
                except Exception:
                    pass

        item_text = clean_text(item_text)
        if not item_text and elem_type not in ["image", "table"]:
            continue

        elem_record = {
            "document_id": pdf_stem,
            "element_id": f"e_{elem_count:04d}",
            "type": elem_type,
            "text": item_text,
            "page": page_no,
            "level": getattr(item, "level", 1) if elem_type == "heading" else None,
            "table_data": table_data
        }

        if elem_type == "image" and image_info:
            elem_record["image_id"] = image_info["image_id"]
            elem_record["image_path"] = image_info["image_path"]
            elem_record["width"] = image_info["width"]
            elem_record["height"] = image_info["height"]
            elem_record["caption"] = image_info["caption"]
            elem_record["ai_caption"] = ""

        elements.append(elem_record)

    return elements, image_records


def convert_pdfs(
    input_path: Path,
    output_path: Path,
    elements_output_path: Path = None,
    *,
    images_dir: Path | None = None,
    image_path_prefix: str | None = None,
):
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(list(input_path.glob("*.pdf")))
        if not pdf_files:
            print(f"⚠ No PDF files found in directory: {input_path}")
            return
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    all_elements: List[Dict[str, Any]] = []
    all_image_records: List[Dict[str, Any]] = []
    all_page_records: List[Dict[str, Any]] = []

    converter = create_converter()

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        pdf_stem = pdf_path.stem
        try:
            result = converter.convert(str(pdf_path))
            document = result.document

            # Default remains data/images/<document_id> for the legacy CLI.
            document_images_dir = (
                images_dir
                if images_dir is not None and len(pdf_files) == 1
                else (images_dir / pdf_stem if images_dir is not None else output_path.parent.parent / "images" / pdf_stem)
            )
            document_image_prefix = (
                image_path_prefix
                if image_path_prefix is not None and len(pdf_files) == 1
                else (
                    f"{image_path_prefix.rstrip('/')}/{pdf_stem}"
                    if image_path_prefix is not None
                    else None
                )
            )

            # 1. Trích xuất các elements chi tiết và lưu file PNG thực tế
            elements, image_records = extract_elements_from_docling(
                document,
                pdf_stem,
                document_images_dir,
                document_image_prefix,
            )
            print(f"  ✓ Extracted {len(elements)} elements & {len(image_records)} physical PNG images")

            all_elements.extend(elements)
            all_image_records.extend(image_records)

            # 2. Gom nhóm theo trang cho file pdf_extract.jsonl
            pages_dict: Dict[int, List[str]] = {}
            for elem in elements:
                p = elem["page"]
                if p not in pages_dict:
                    pages_dict[p] = []
                if elem["text"]:
                    pages_dict[p].append(elem["text"])

            for page_no in sorted(pages_dict.keys()):
                page_text = "\n\n".join(pages_dict[page_no])
                record = {
                    "id": f"{pdf_stem}_page_{page_no}",
                    "content": page_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_no,
                        "title": pdf_stem,
                        "file_type": "pdf"
                    }
                }
                all_page_records.append(record)
        except Exception as exc:
            print(f"❌ Failed to process {pdf_path.name}: {exc}")

    if elements_output_path is None:
        elements_output_path = output_path.parent / "elements.jsonl"

    elements_output_path.parent.mkdir(parents=True, exist_ok=True)
    with elements_output_path.open("w", encoding="utf-8") as f:
        for elem in all_elements:
            f.write(json.dumps(elem, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved total elements ({len(all_elements)}): {elements_output_path}")

    # Ghi file image_metadata.jsonl làm reference
    metadata_path = output_path.parent / "image_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in all_image_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved total image metadata ({len(all_image_records)}): {metadata_path}")

    # 3. Tạo file pdf_extract.jsonl với tất cả các trang của mọi PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_page_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Saved pdf_extract.jsonl ({len(all_page_records)} pages): {output_path}")

    print("=" * 60)
    print("PDF PARSE COMPLETE")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Elements: {elements_output_path}")
    print(f"PDF Extract: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF(s) into JSONL Elements for RAG")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/processed/pdf_extract.jsonl"), type=Path)
    parser.add_argument("--elements-output", default=Path("data/processed/elements.jsonl"), type=Path)

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    convert_pdfs(args.input, args.output, args.elements_output)


if __name__ == "__main__":
    main()
