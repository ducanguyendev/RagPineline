"""Import the reviewed legal evaluation worksheet into the active JSON dataset."""

import argparse
import os
import json
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from openpyxl import load_workbook
except ImportError:  # use a small standard-library reader when offline
    load_workbook = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHEET = "Luật LĐ"
METADATA_COLUMNS = {
    "category": ("category",),
    "source_page": ("source page", "page"),
    "source_section": ("source section", "section"),
    "required_keywords": ("required keywords", "key concepts", "required keywords / key concepts"),
    "retrieval_difficulty": ("retrieval difficulty", "difficulty"),
    "expected_retrieval_type": ("expected retrieval type", "retrieval type"),
    "is_answerable": ("is answerable", "answerable"),
    "expected_behavior": ("expected behavior",),
    "notes": ("notes", "note"),
}


def _normalize_header(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value).strip() or None


def _answerable(value: Any, testcase_id: Any) -> Any:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(f"Test case {testcase_id}: unrecognized Is Answerable value {value!r}")


def _find_column(headers: dict[str, int], *aliases: str) -> int | None:
    normalized_aliases = {_normalize_header(alias) for alias in aliases}
    for header, index in headers.items():
        if header in normalized_aliases:
            return index
    return None


def _resolve_evidence_column(headers: dict[str, int], raw_headers: tuple[Any, ...], data_rows: list[tuple[Any, ...]], answer_column: int, page_column: int | None) -> int:
    """Find named Evidence, or verify the blank header between answer and page."""
    named = _find_column(headers, "evidence", "reference evidence")
    if named is not None:
        return named
    if page_column is None or page_column != answer_column + 2:
        raise ValueError("Cannot safely identify Evidence: no Evidence header and Source Page is not exactly two columns after Expected Answer")
    candidate = answer_column + 1
    if candidate >= len(raw_headers) or _clean(raw_headers[candidate]) is not None:
        raise ValueError("Cannot safely identify Evidence: the column between Expected Answer and Source Page does not have a blank header")
    nonempty = 0
    page_like = 0
    for row in data_rows:
        if candidate >= len(row):
            continue
        value = _clean(row[candidate])
        if value is None:
            continue
        nonempty += 1
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            page_like += 1
    if not nonempty:
        raise ValueError("Cannot safely identify Evidence: the blank-header candidate column contains no data")
    if page_like:
        raise ValueError(f"Cannot safely identify Evidence: candidate column {candidate + 1} contains {page_like} numeric value(s), so it may be page data")
    return candidate


def _xlsx_rows_stdlib(input_path: Path, sheet_name: str) -> list[tuple[Any, ...]]:
    """Read cell values for one worksheet from OOXML without external packages."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships", "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(input_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("p:Relationship", ns)}
        sheet_path = None
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                target = targets[sheet.attrib[f"{{{ns['r']}}}id"]]
                sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
                break
        if sheet_path is None:
            names = [sheet.attrib.get("name", "") for sheet in workbook.findall("m:sheets/m:sheet", ns)]
            raise ValueError(f"Worksheet {sheet_name!r} not found. Available: {names}")
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(part.text or "" for part in item.findall(".//m:t", ns)) for item in shared_root.findall("m:si", ns)]
        root = ET.fromstring(archive.read(sheet_path))
        parsed = []
        max_column = 0
        for row in root.findall("m:sheetData/m:row", ns):
            values = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                col_letters = re.match(r"[A-Z]+", ref)
                if not col_letters:
                    continue
                col = 0
                for char in col_letters.group(0):
                    col = col * 26 + ord(char) - 64
                max_column = max(max_column, col)
                value_node = cell.find("m:v", ns)
                value = value_node.text if value_node is not None else None
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(part.text or "" for part in cell.findall(".//m:t", ns))
                elif cell.attrib.get("t") == "b" and value is not None:
                    value = value == "1"
                elif value is not None:
                    try:
                        number = float(value)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        pass
                values[col] = value
            parsed.append(tuple(values.get(col) for col in range(1, max_column + 1)))
        return parsed


def convert_workbook(input_path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if load_workbook is None:
        all_rows = _xlsx_rows_stdlib(input_path, sheet_name)
    else:
        workbook = load_workbook(input_path, read_only=True, data_only=True)
        try:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"Worksheet {sheet_name!r} not found. Available: {workbook.sheetnames}")
            all_rows = list(workbook[sheet_name].iter_rows(values_only=True))
        finally:
            workbook.close()
    if not all_rows:
        raise ValueError(f"Worksheet {sheet_name!r} is empty")
    raw_headers, data_rows = all_rows[0], all_rows[1:]
    headers = {_normalize_header(value): i for i, value in enumerate(raw_headers) if value is not None}
    required_columns = {
        "id": _find_column(headers, "test case id", "testcase id", "id"),
        "query": _find_column(headers, "query", "question"),
        "expected_answer": _find_column(headers, "expected answer", "expected response", "ground truth"),
    }
    missing = [name for name, index in required_columns.items() if index is None]
    if missing:
        raise ValueError(f"Worksheet is missing required column(s): {', '.join(missing)}")
    page_column = _find_column(headers, *METADATA_COLUMNS["source_page"])
    evidence_column = _resolve_evidence_column(headers, raw_headers, data_rows, required_columns["expected_answer"], page_column)
    metadata_columns = {key: _find_column(headers, *aliases) for key, aliases in METADATA_COLUMNS.items()}
    dataset = []
    missing_rows = []
    for excel_row, row in enumerate(data_rows, start=2):
        if not any(_clean(value) is not None for value in row):
            continue
        testcase_id = _clean(row[required_columns["id"]])
        query = _clean(row[required_columns["query"]])
        answer = _clean(row[required_columns["expected_answer"]])
        if not testcase_id or not query or not answer:
            missing_rows.append((excel_row, testcase_id, not bool(query), not bool(answer)))
            continue
        evidence = _clean(row[evidence_column]) if evidence_column < len(row) else None
        if evidence is None:
            raise ValueError(f"Test case {testcase_id}: Evidence is empty in Excel row {excel_row}")
        metadata = {}
        for key, column in metadata_columns.items():
            if column is None:
                continue
            value = row[column] if column < len(row) else None
            value = _answerable(value, testcase_id) if key == "is_answerable" else _clean(value)
            metadata[key] = value
        dataset.append({
            "id": str(testcase_id),
            "query": str(query),
            "expected_answer": str(answer),
            "reference_contexts": [str(evidence)],
            "metadata": metadata,
        })
    if missing_rows:
        examples = "; ".join(f"row {row}: id={tid!r}, missing query={q}, answer={a}" for row, tid, q, a in missing_rows[:5])
        raise ValueError(f"{len(missing_rows)} data row(s) lack required values; {examples}")
    ids = [item["id"] for item in dataset]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate testcase IDs: {', '.join(duplicates)}")
    return dataset


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    if not isinstance(dataset, list):
        raise ValueError("Converted dataset root must be an array")
    ids = []
    for index, item in enumerate(dataset, start=1):
        if not all(isinstance(item.get(field), str) and item[field].strip() for field in ("id", "query", "expected_answer")):
            raise ValueError(f"Converted testcase {index} has an empty required field")
        if not isinstance(item.get("reference_contexts"), list):
            raise ValueError(f"Testcase {item['id']} reference_contexts must be an array")
        ids.append(item["id"])
        json.dumps(item, allow_nan=False)
    if len(set(ids)) != len(ids):
        raise ValueError("Converted dataset contains duplicate IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an Excel worksheet into the RAG evaluation dataset")
    parser.add_argument("--input", type=Path, required=True, help="Source .xlsx workbook")
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "eval_dataset.json")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input workbook not found: {args.input}")
    try:
        dataset = convert_workbook(args.input, args.sheet)
        validate_dataset(dataset)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    with temp_output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp_output, args.output)
    evidence_count = sum(bool(item["reference_contexts"]) for item in dataset)
    print(f"Worksheet        : {args.sheet}")
    print(f"Imported cases   : {len(dataset)}")
    print(f"Unique IDs       : {len({item['id'] for item in dataset})}")
    print("Missing query    : 0")
    print("Missing answer   : 0")
    print(f"With evidence    : {evidence_count}")
    print(f"Without evidence : {len(dataset) - evidence_count}")
    print(f"Output           : {args.output}")


if __name__ == "__main__":
    main()
