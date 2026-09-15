"""
Image Captioner for RAG multimodal pipeline (Optimized)

Source of Truth: elements.jsonl

1. Image Filter: Skip small/decorative images (< 100x100 px).
2. BLIP AI Captioning: Generate semantic description.
3. Enrich: Update `ai_caption` directly in elements.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_AREA = 10000


def load_captioner(model_name: str = "Salesforce/blip-image-captioning-base"):
    print(f"Loading BLIP model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    print("✓ Model loaded")
    return processor, model


def generate_caption(
    image_path: Path,
    processor,
    model,
    max_length: int = 50
) -> str:
    try:
        if not image_path.exists():
            return ""
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"✗ Failed to caption {image_path}: {e}")
        return ""


def caption_elements(
    elements_path: Path,
    output_path: Path = None,
    model_name: str = "Salesforce/blip-image-captioning-base"
):
    if not elements_path.exists():
        raise FileNotFoundError(f"Elements file not found: {elements_path}")

    print(f"Loading elements: {elements_path}")
    elements: List[Dict[str, Any]] = []
    with elements_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                elements.append(json.loads(line))

    # Tìm các element dạng image có image_path
    image_elements = [
        (idx, elem) for idx, elem in enumerate(elements)
        if elem.get("type") == "image" and elem.get("image_path")
    ]

    print(f"Found {len(image_elements)} image elements")

    if not image_elements:
        print("No image elements to process.")
        return

    processor, model = load_captioner(model_name)

    caption_count = 0
    skip_count = 0

    for idx, elem in image_elements:
        rel_path = elem.get("image_path", "")
        image_id = elem.get("image_id", f"img_{idx}")
        abs_path = ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)

        width = elem.get("width", 0)
        height = elem.get("height", 0)

        # Lấy kích thước thực của ảnh nếu chưa có
        if (width == 0 or height == 0) and abs_path.exists():
            try:
                with Image.open(abs_path) as img:
                    width, height = img.size
                    elem["width"] = width
                    elem["height"] = height
            except Exception:
                pass

        # IMAGE FILTER: Bỏ qua ảnh rác / trang trí nhỏ (< 100x100 px)
        if width < MIN_WIDTH or height < MIN_HEIGHT or (width * height) < MIN_AREA:
            print(f"⏩ [Skip Filter] {image_id} ({width}x{height} px too small)")
            skip_count += 1
            elem["ai_caption"] = ""
            continue

        print(f"🖼️ [{caption_count + 1}] Captioning {image_id} ({width}x{height} px)...")
        ai_caption = generate_caption(abs_path, processor, model)
        elem["ai_caption"] = ai_caption
        caption_count += 1
        print(f"   ↳ Result: '{ai_caption}'")

    # Lưu cập nhật trực tiếp lại elements.jsonl (Source of Truth)
    save_path = output_path or elements_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        for elem in elements:
            f.write(json.dumps(elem, ensure_ascii=False) + "\n")

    # Xuất file cache image_captions.jsonl để debug
    cache_path = elements_path.parent / "image_captions.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for idx, elem in image_elements:
            if elem.get("ai_caption"):
                f.write(json.dumps({
                    "document_id": elem.get("document_id"),
                    "image_id": elem.get("image_id"),
                    "image_path": elem.get("image_path"),
                    "ai_caption": elem.get("ai_caption")
                }, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("ENRICH IMAGE ELEMENTS COMPLETE")
    print(f"Captioned: {caption_count} | Skipped: {skip_count}")
    print(f"Updated Source of Truth: {save_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate AI captions for image elements in elements.jsonl")
    parser.add_argument("--elements", type=Path, default=Path("data/processed/elements.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=str, default="Salesforce/blip-image-captioning-base")

    args = parser.parse_args()
    caption_elements(args.elements, args.output, args.model)


if __name__ == "__main__":
    main()
