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
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import re
import shutil
import subprocess
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel import pipeline_options as pipeline_options_module
from docling.datamodel.pipeline_options import PdfPipelineOptions
from tqdm import tqdm
from src.data.pdf_errors import PdfInputError, PdfRuntimeError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOGGER = logging.getLogger(__name__)


def _docling_version() -> str:
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _accelerator_options():
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Docling {_docling_version()} does not expose accelerator options."
        ) from exc
    return AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU,
    )


def _resolve_tesseract() -> str | None:
    executable = shutil.which("tesseract")
    if executable:
        return executable
    common_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if common_windows_path.is_file():
        return str(common_windows_path)
    return None


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as HH:MM:SS, including runs longer than a day."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def docling_heartbeat(pdf_name: str, interval: float = 10.0):
    """Report elapsed time only; Docling does not expose per-PDF progress here."""
    stop_event = Event()
    started_at = perf_counter()

    def report_elapsed():
        while not stop_event.wait(interval):
            tqdm.write(
                f"⏳ Docling processing {pdf_name} | "
                f"elapsed {format_elapsed(perf_counter() - started_at)}",
                file=sys.stdout,
            )

    heartbeat = Thread(target=report_elapsed, name="docling-heartbeat", daemon=True)
    heartbeat.start()
    try:
        yield started_at
    finally:
        stop_event.set()
        heartbeat.join()


def clean_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tesseract_languages() -> set[str]:
    executable = _resolve_tesseract()
    if not executable:
        return set()
    try:
        completed = subprocess.run(
            [executable, "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available")
    }


def create_converter(*, enable_ocr: bool = True):
    pipeline_options = PdfPipelineOptions()
    try:
        pipeline_options.accelerator_options = _accelerator_options()
    except Exception as exc:
        LOGGER.exception("Unable to configure Docling CPU accelerator")
        raise PdfRuntimeError(
            f"Docling {_docling_version()} could not be configured for CPU conversion. "
            "See backend logs for details."
        ) from exc
    if enable_ocr:
        cli_options_cls = getattr(
            pipeline_options_module, "TesseractCliOcrOptions", None
        )
        binding_options_cls = getattr(
            pipeline_options_module, "TesseractOcrOptions", None
        )
        ocr_options_cls = cli_options_cls
        if ocr_options_cls is None:
            binding_available = importlib.util.find_spec("tesserocr") is not None
            if binding_available:
                ocr_options_cls = binding_options_cls
            else:
                raise PdfInputError(
                    f"OCR failed for this PDF: Docling {_docling_version()} has no "
                    "Tesseract CLI backend and the tesserocr Python binding is unavailable."
                )
        languages = _tesseract_languages()
        if not {"vie", "eng"}.issubset(languages):
            raise PdfInputError(
                "OCR is required for this PDF but Tesseract Vietnamese language "
                "data is unavailable (required languages: vie, eng)."
            )
        option_kwargs = {}
        try:
            parameters = inspect.signature(ocr_options_cls).parameters
        except (TypeError, ValueError):
            parameters = {}
        if cli_options_cls is not None and "tesseract_cmd" in parameters:
            option_kwargs["tesseract_cmd"] = _resolve_tesseract() or "tesseract"
        if "lang" in parameters or not parameters:
            option_kwargs["lang"] = ["vie", "eng"]
        else:
            option_kwargs["languages"] = ["vie", "eng"]
        try:
            pipeline_options.ocr_options = ocr_options_cls(**option_kwargs)
        except Exception as exc:
            LOGGER.exception("Unable to configure Docling Tesseract OCR backend")
            raise PdfRuntimeError(
                f"Docling {_docling_version()} could not configure Tesseract OCR. "
                "See backend logs for details."
            ) from exc
        LOGGER.info(
            "Docling %s OCR backend=%s executable=%s languages=vie,eng",
            _docling_version(),
            "tesseract-cli" if cli_options_cls is not None else "tesserocr-binding",
            _resolve_tesseract(),
        )
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _ocr_backend_name() -> str:
    if getattr(pipeline_options_module, "TesseractCliOcrOptions", None) is not None:
        return "tesseract-cli"
    if getattr(pipeline_options_module, "TesseractOcrOptions", None) is not None:
        return "tesserocr-binding"
    return "unavailable"


def _document_diagnostics(document, elements: list[dict[str, Any]]) -> dict[str, int]:
    pages = getattr(getattr(document, "pages", None), "keys", lambda: [])()
    page_count = len(pages) if pages else len(getattr(document, "pages", []) or [])
    text_count = sum(1 for element in elements if element.get("text"))
    image_count = sum(1 for element in elements if element.get("type") == "image")
    table_count = sum(1 for element in elements if element.get("type") == "table")
    return {
        "pages": page_count,
        "elements": len(elements),
        "text_elements": text_count,
        "images": image_count,
        "tables": table_count,
    }


def _needs_ocr(document, elements: list[dict[str, Any]]) -> bool:
    pages = getattr(document, "pages", None)
    page_numbers = set(pages.keys()) if hasattr(pages, "keys") else set()
    if not page_numbers:
        page_numbers = {int(element.get("page") or 1) for element in elements}
    text_pages = {
        int(element.get("page") or 1)
        for element in elements
        if element.get("text")
    }
    return not text_pages or bool(page_numbers - text_pages)


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
                    tqdm.write(f"! Failed to extract image {image_id}: {exc}", file=sys.stdout)

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

    total_started_at = perf_counter()
    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    all_elements: List[Dict[str, Any]] = []
    all_image_records: List[Dict[str, Any]] = []
    all_page_records: List[Dict[str, Any]] = []

    native_converter = create_converter(enable_ocr=False)
    converter = None

    total_pdfs = len(pdf_files)
    successful_pdfs = 0
    failed_pdfs = 0
    with tqdm(total=total_pdfs, desc=f"PDF 0/{total_pdfs}", unit="pdf", dynamic_ncols=True, file=sys.stdout) as progress:
        for pdf_index, pdf_path in enumerate(pdf_files, start=1):
            pdf_started_at = perf_counter()
            progress.set_postfix_str(f"Processing: {pdf_path.name}")
            tqdm.write("\n" + "=" * 70, file=sys.stdout)
            tqdm.write(f"[{pdf_index}/{total_pdfs}] Processing: {pdf_path.name}", file=sys.stdout)
            tqdm.write("=" * 70, file=sys.stdout)
            pdf_stem = pdf_path.stem
            ocr_used = False
            try:
                with docling_heartbeat(pdf_path.name) as docling_started_at:
                    result = native_converter.convert(str(pdf_path))
                LOGGER.info(
                    "PDF: %s | Docling: %s | Docling device: cpu | Native parse: success",
                    pdf_path.name,
                    _docling_version(),
                )
                tqdm.write(
                    f"✓ Docling completed in {(perf_counter() - docling_started_at) / 60:.2f} minutes",
                    file=sys.stdout,
                )
                document = result.document

                # Thư mục lưu ảnh: data/images/<document_id>/
                document_images_dir = (
                    images_dir
                    if images_dir is not None and len(pdf_files) == 1
                    else (
                        images_dir / pdf_stem
                        if images_dir is not None
                        else output_path.parent.parent / "images" / pdf_stem
                    )
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
                if _needs_ocr(document, elements):
                    tqdm.write(
                        f"Native text missing on one or more pages in {pdf_path.name}; "
                        "retrying with hybrid OCR.",
                        file=sys.stdout,
                    )
                    if converter is None:
                        try:
                            converter = create_converter(enable_ocr=True)
                        except PdfInputError as exc:
                            raise PdfInputError(
                                f"OCR failed for {pdf_path.name}: {exc}"
                            ) from exc
                    ocr_used = True
                    with docling_heartbeat(pdf_path.name + " (OCR)"):
                        result = converter.convert(str(pdf_path))
                    document = result.document
                    elements, image_records = extract_elements_from_docling(
                        document,
                        pdf_stem,
                        document_images_dir,
                        document_image_prefix,
                    )
                diagnostics = _document_diagnostics(document, elements)
                diagnostic_line = (
                    f"PDF: {pdf_path.name} | Docling device: cpu | "
                    f"Native parse: success | OCR fallback: {str(ocr_used).lower()} | "
                    f"OCR engine: {_ocr_backend_name() if ocr_used else 'none'} | "
                    f"OCR languages: {'vie, eng' if ocr_used else 'none'} | "
                    f"Pages: {diagnostics['pages']} | Elements: {diagnostics['elements']} | "
                    f"Text elements: {diagnostics['text_elements']} | "
                    f"Tables: {diagnostics['tables']} | Images: {diagnostics['images']}"
                )
                tqdm.write(diagnostic_line, file=sys.stdout)
                LOGGER.info(
                    "%s | Docling version: %s",
                    diagnostic_line,
                    _docling_version(),
                )
                if not elements:
                    raise PdfInputError(
                        f"Docling produced zero elements for {pdf_path.name} "
                        f"(pages={diagnostics['pages']}, elements=0, "
                        f"text_elements=0, images=0, tables=0, "
                        f"ocr_enabled={ocr_used})."
                    )
                tqdm.write(
                    f"✓ Extracted {len(elements)} elements & {len(image_records)} physical PNG images",
                    file=sys.stdout,
                )

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
                failed_pdfs += 1
                tqdm.write(f"❌ Failed [{pdf_index}/{total_pdfs}]: {pdf_path.name}", file=sys.stdout)
                tqdm.write(f"Reason: {exc}", file=sys.stdout)
                if isinstance(exc, (PdfInputError, PdfRuntimeError)):
                    raise
                LOGGER.exception(
                    "PDF processing failed: file=%s docling=%s device=cpu "
                    "native_parse=failure ocr_fallback=%s",
                    pdf_path.name,
                    _docling_version(),
                    ocr_used,
                )
                error_text = str(exc).lower()
                if "fp64" in error_text or "not supported on the device" in error_text:
                    raise PdfRuntimeError(
                        "PDF conversion failed while running Docling. Docling was "
                        "configured for CPU but conversion still failed. "
                        "See backend logs for details."
                    ) from exc
                if ocr_used:
                    raise PdfRuntimeError(
                        f"OCR failed for {pdf_path.name}: Tesseract CLI conversion "
                        "failed. See backend logs for details."
                    ) from exc
                raise PdfInputError(
                    f"Could not read or parse PDF {pdf_path.name}. "
                    "Check that the file is valid."
                ) from exc
            else:
                successful_pdfs += 1
                tqdm.write(f"✓ Completed [{pdf_index}/{total_pdfs}]", file=sys.stdout)

            tqdm.write(f"Elapsed for PDF : {format_elapsed(perf_counter() - pdf_started_at)}", file=sys.stdout)
            tqdm.write(f"Total elapsed   : {format_elapsed(perf_counter() - total_started_at)}", file=sys.stdout)
            # Advance only after this PDF has finished processing (success or failure).
            progress.set_description(f"PDF {successful_pdfs + failed_pdfs}/{total_pdfs}", refresh=False)
            progress.set_postfix_str(f"Successful: {successful_pdfs}, Failed: {failed_pdfs}", refresh=False)
            progress.update(1)

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
    print(f"Successful PDFs : {successful_pdfs}")
    print(f"Failed PDFs     : {failed_pdfs}")
    print(f"Total elapsed   : {format_elapsed(perf_counter() - total_started_at)}")
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
