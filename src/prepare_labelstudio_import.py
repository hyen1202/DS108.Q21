"""
prepare_labelstudio_import_vecner_v4_2.py
=========================================
Chuyển CSV/JSON/JSONL tiêu đề sản phẩm TMĐT sang task import cho Label Studio.

- Không sửa file đầu vào.
- Giữ nguyên nội dung text ngoài chuẩn hóa Unicode NFC.
- Tạo labels_for_task động theo ngành hàng.
- Nhãn O không xuất hiện trên giao diện: annotator không bôi đen = O.
- Có flag task-level WRONG_INDUSTRY và NEEDS_ADJUDICATION trong XML config.

Usage:
    python prepare_labelstudio_import_vecner_v4_2.py --input products.csv --output ls_tasks.json
    python prepare_labelstudio_import_vecner_v4_2.py --input products.jsonl --output ls_tasks.jsonl
"""

import argparse
import csv
import json
import logging
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: Any) -> str:
    """Chỉ chuẩn hóa Unicode NFC, không strip/lower/clean để giữ nguyên title gốc."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFC", text)


def _iter_sources(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = [item]
    data = item.get("data")
    meta = item.get("meta")
    if isinstance(data, dict):
        sources.append(data)
        if isinstance(data.get("meta"), dict):
            sources.append(data["meta"])
    if isinstance(meta, dict):
        sources.append(meta)
    return sources


def first_nonempty(item: Dict[str, Any], keys: List[str]) -> str:
    """Lấy giá trị đầu tiên không rỗng từ item/data/meta."""
    for source in _iter_sources(item):
        for key in keys:
            val = source.get(key)
            if val not in (None, ""):
                return str(val)
    return ""


def extract_text(item: Dict[str, Any]) -> str:
    """
    Ưu tiên title_clean nếu pipeline đã có, nhưng không biến đổi thêm.
    Có thể đổi thứ tự keys nếu nhóm muốn dùng title_raw/title gốc tuyệt đối.
    """
    return first_nonempty(item, ["title_clean", "text", "title", "title_raw", "product_title"])


def extract_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Đưa metadata vào data.meta để tham khảo/audit, không dùng làm text gán nhãn."""
    meta: Dict[str, Any] = {}
    meta_keys = [
        "id", "product_id", "platform", "category_l1", "category_l2", "category_l3",
        "industry_group", "price", "product_url", "source_file", "brand", "brand_clean",
        "catalog_brand",
    ]
    for source in _iter_sources(item):
        for key in meta_keys:
            if key in source and source[key] not in (None, "") and key not in meta:
                meta[key] = source[key]
    if "catalog_brand" not in meta:
        for k in ("brand_clean", "brand"):
            if meta.get(k) not in (None, "", "no brand", "No Brand"):
                meta["catalog_brand"] = meta[k]
                break
    return meta


def extract_context(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "category_l1": first_nonempty(item, ["category_l1"]),
        "category_l2": first_nonempty(item, ["category_l2"]),
        "category_l3": first_nonempty(item, ["category_l3"]),
        "industry_group": first_nonempty(item, ["industry_group"]),
        "catalog_brand": first_nonempty(item, ["catalog_brand", "brand_clean", "brand"]),
    }


# Theo guideline VEC-NER mới:
# - O không đưa vào Label Studio vì O = không bôi đen.
# - TEXTURE bị bỏ vì guideline không còn định nghĩa nhãn này.
COMMON_LABELS = [
    "PRODUCT_TYPE", "BRAND", "MODEL", "QUANTITY", "ORIGIN", "COMPONENT",
    "MATERIAL", "SIZE", "COMPAT", "ATTRIBUTE", "OCCASION", "EFFECT",
    "COLOR", "STYLE", "TARGET_GROUP",
]

DOMAIN_LABELS: Dict[str, List[str]] = {
    "electronics": ["SPEC", "CONNECTIVITY"],
    "home_living": ["POWER", "CAPACITY"],
    "fashion": [],
    "beauty_health": ["INGREDIENT", "SKIN_TYPE", "BODY_PART", "VOLUME_WEIGHT", "CONCENTRATION"],
}

# Thứ tự ưu tiên để 9 nhãn đầu dễ bấm hotkey.
LABELS_ORDERED: Dict[str, List[str]] = {
    "electronics": [
        "PRODUCT_TYPE", "BRAND", "MODEL", "SPEC", "CONNECTIVITY",
        "COMPAT", "ATTRIBUTE", "QUANTITY", "ORIGIN",
        "MATERIAL", "SIZE", "COLOR", "STYLE", "OCCASION", "EFFECT",
        "TARGET_GROUP", "COMPONENT",
    ],
    "home_living": [
        "PRODUCT_TYPE", "BRAND", "POWER", "CAPACITY", "ATTRIBUTE",
        "MATERIAL", "SIZE", "COLOR", "OCCASION",
        "QUANTITY", "ORIGIN", "MODEL", "COMPONENT", "COMPAT", "EFFECT",
        "STYLE", "TARGET_GROUP",
    ],
    "fashion": [
        "PRODUCT_TYPE", "BRAND", "TARGET_GROUP", "STYLE", "SIZE",
        "COLOR", "MATERIAL", "OCCASION", "ATTRIBUTE",
        "QUANTITY", "ORIGIN", "MODEL", "COMPONENT", "COMPAT", "EFFECT",
    ],
    "beauty_health": [
        "PRODUCT_TYPE", "EFFECT", "INGREDIENT", "BRAND", "BODY_PART",
        "SKIN_TYPE", "VOLUME_WEIGHT", "CONCENTRATION", "ATTRIBUTE",
        "OCCASION", "QUANTITY", "ORIGIN", "MODEL", "COMPONENT", "TARGET_GROUP",
        "COLOR", "STYLE", "MATERIAL", "SIZE", "COMPAT",
    ],
}

ALL_LABELS = []
for label in COMMON_LABELS + DOMAIN_LABELS["electronics"] + DOMAIN_LABELS["home_living"] + DOMAIN_LABELS["beauty_health"]:
    if label not in ALL_LABELS:
        ALL_LABELS.append(label)

INDUSTRY_MAP = {
    "electronics": "electronics", "electrics": "electronics", "electronics & tech": "electronics",
    "tech": "electronics", "electronic": "electronics", "điện tử": "electronics",
    "home & living": "home_living", "home": "home_living", "living": "home_living",
    "home_living": "home_living", "gia dụng": "home_living", "nhà cửa": "home_living",
    "fashion": "fashion", "fashion & accessories": "fashion", "accessories": "fashion",
    "thời trang": "fashion",
    "beauty": "beauty_health", "health": "beauty_health", "beauty & health": "beauty_health",
    "health & beauty": "beauty_health", "beauty_health": "beauty_health", "mỹ phẩm": "beauty_health",
}


def infer_industry_key(industry: str) -> Optional[str]:
    raw = (industry or "").lower().strip()
    if not raw:
        return None
    if raw in INDUSTRY_MAP:
        return INDUSTRY_MAP[raw]
    for alias, key in INDUSTRY_MAP.items():
        if alias in raw or raw in alias:
            return key
    return None


def build_label_set(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Nhãn chính theo ngành được đưa lên trước.
    Các nhãn còn lại được thêm cuối dưới dạng fallback xám để xử lý mẫu sai ngành.
    """
    ctx = extract_context(item)
    industry_key = infer_industry_key(ctx.get("industry_group", ""))
    primary = LABELS_ORDERED.get(industry_key or "", ALL_LABELS)

    final_order: List[str] = []
    for label in primary + ALL_LABELS:
        if label not in final_order:
            final_order.append(label)

    primary_set = set(primary)
    formatted: List[Dict[str, str]] = []
    for label in final_order:
        if label in primary_set or industry_key is None:
            formatted.append({"value": label})
        else:
            formatted.append({
                "value": label,
                "html": f"<span style='color:#6c757d; font-size:11px; font-weight:normal;'>💤 {label}</span>",
                "background": "#f1f3f5",
            })
    return formatted


def build_hotkey_hint(labels_for_task: List[Dict[str, str]]) -> str:
    labels = [x["value"] for x in labels_for_task[:9]]
    return "  ".join(f"{i + 1}:{label}" for i, label in enumerate(labels))


def build_context_header(ctx: Dict[str, str]) -> str:
    industry = (ctx.get("industry_group") or "UNKNOWN").upper()
    breadcrumb = " › ".join([ctx.get("category_l1", ""), ctx.get("category_l2", ""), ctx.get("category_l3", "")]).strip(" ›")
    header = f"📁 [{industry}]"
    if breadcrumb:
        header += f"  {breadcrumb}"
    brand = ctx.get("catalog_brand")
    if brand and brand not in ("no brand", "No Brand"):
        header += f"  |  🏷️ Brand: {brand}"
    return header


def load_items(input_path: Path) -> List[Dict[str, Any]]:
    ext = input_path.suffix.lower()
    if ext == ".csv":
        with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("File JSON phải chứa một mảng object.")
        return data
    if ext == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Định dạng không hỗ trợ: {ext}. Dùng .csv, .json hoặc .jsonl")


def choose_task_id(item: Dict[str, Any]) -> str:
    for key in ("id", "product_id"):
        val = first_nonempty(item, [key])
        if val:
            return val
    return str(uuid.uuid4())


def process_file(input_path: Path, output_path: Path) -> None:
    items = load_items(input_path)
    logger.info("Đã đọc %s mẫu từ %s", len(items), input_path.name)

    tasks: List[Dict[str, Any]] = []
    skipped = 0
    for item in items:
        raw_text = extract_text(item)
        if raw_text == "":
            skipped += 1
            continue

        text = normalize_text(raw_text)
        ctx = extract_context(item)
        meta = extract_metadata(item)
        labels_for_task = build_label_set(item)

        tasks.append({
            "id": choose_task_id(item),
            "data": {
                "text": text,
                "context_header": build_context_header(ctx),
                "industry_group": ctx["industry_group"],
                "catalog_brand": ctx["catalog_brand"],
                "category_l1": ctx["category_l1"],
                "category_l2": ctx["category_l2"],
                "category_l3": ctx["category_l3"],
                "labels_for_task": labels_for_task,
                "hotkey_hint": build_hotkey_hint(labels_for_task),
                "meta": meta,
            },
        })

    with open(output_path, "w", encoding="utf-8") as f:
        if output_path.suffix.lower() == ".json":
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        else:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.info("Hoàn tất. input=%s output=%s skipped=%s", len(items), len(tasks), skipped)
    logger.info("File xuất: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Label Studio import tasks for VEC-NER.")
    parser.add_argument("--input", "-i", required=True, help="Input .csv/.json/.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Output .json hoặc .jsonl")
    args = parser.parse_args()
    process_file(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
