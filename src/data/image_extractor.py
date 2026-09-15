"""
Extract images from PDF for RAG multimodal pipeline.

Input:
    PDF

Output:
    images/*.png
    image_metadata.jsonl
"""

import os

# Disable torch compile (Triton issues on Windows)
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions



def extract_images(
    pdf_path: Path,
    output_dir: Path,
    metadata_path: Path
):

    print(f"Processing: {pdf_path}")

    # Configure pipeline - minimal options to avoid Triton issues
    pipeline_options = PdfPipelineOptions()
    
    # Disable OCR (PDF has text layer)
    pipeline_options.do_ocr = False
    
    # Keep table structure
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


    result = converter.convert(
        str(pdf_path)
    )


    doc = result.document


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    image_records = []

    image_count = 0

    # Track different item types for debugging
    item_type_count = {}

    # Docling iterate toàn bộ item
    for item, _level in doc.iterate_items():

        # Track all types
        item_class = item.__class__.__name__
        item_type_count[item_class] = item_type_count.get(item_class, 0) + 1

        # Look for images - check multiple possible class names
        is_image = False
        
        if item_class.lower() == "picture":
            is_image = True
        elif "image" in item_class.lower():
            is_image = True
        elif "figure" in item_class.lower():
            is_image = True
        elif "illustration" in item_class.lower():
            is_image = True
        elif hasattr(item, "get_image"):
            # If it has get_image method, it's likely an image
            is_image = True

        if not is_image:
            continue

        image_count += 1


        page_no = 1


        if hasattr(item, "prov") and item.prov:

            page_no = item.prov[0].page_no



        image_id = (
            f"page_{page_no:03d}"
            f"_img_{image_count:03d}"
        )


        image_path = (
            output_dir
            /
            f"{image_id}.png"
        )


        try:

            image = item.get_image(
                doc
            )


            image.save(
                image_path
            )


        except Exception as e:

            print(
                f"Skip image {image_id}: {e}"
            )

            continue



        image_records.append(

            {
                "image_id":
                    image_id,

                "source":
                    pdf_path.name,

                "page":
                    page_no,

                "image_path":
                    str(image_path)

            }

        )


    # Write metadata even if no images found
    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in image_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                +
                "\n"
            )

    # Print debug info
    if len(item_type_count) > 0 and image_count == 0:
        print(
            "\nℹ PDF item types found:"
        )
        for itype, count in sorted(
            item_type_count.items(),
            key=lambda x: -x[1]
        ):
            if count > 0:
                print(
                    f"  {itype}: {count}"
                )
        print(
            "\nℹ This PDF appears to be text-only"
        )

    print("=" * 60)
    print("DONE")
    print(
        f"Images extracted: {len(image_records)}"
    )
    print(
        f"Image folder: {output_dir}"
    )
    print(
        f"Metadata: {metadata_path}"
    )
    print("=" * 60)



def main():

    parser = argparse.ArgumentParser(
        description="Extract PDF images for RAG"
    )


    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/images"
        )
    )


    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/image_metadata.jsonl"
        )
    )


    args = parser.parse_args()


    extract_images(
        args.input,
        args.output_dir,
        args.metadata
    )



if __name__ == "__main__":
    main()