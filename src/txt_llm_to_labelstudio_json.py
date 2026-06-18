"""
txt_llm_to_labelstudio_json.py  —  v2.1
=========================================
Convert LLM TXT NER output → Label Studio JSON (predictions format).

KIẾN TRÚC:
  - CSV là source of truth: data.text = csv_text RAW (không NFC, không strip).
  - LLM chỉ cung cấp (label, span_text).
  - Offset map trên csv_text.
  - Nếu LLM sửa text → need_adjudication=True, KHÔNG fallback.

INPUT TXT FORMAT (hỗ trợ cả 2 dạng):
  Single-line:  [ID: lazada_001] <BRAND>Apple</BRAND> iPhone 13
  Multi-line:   [ID: lazada_001]
                <BRAND>Apple</BRAND>
                iPhone 13

USAGE:
  python txt_llm_to_labelstudio_json.py \\
      --input  part1.txt --csv campaign.csv --output part1_ls.json \\
      --model-version claude-sonnet-4-20250514 \\
      --warn-file warnings.csv [--strict-copy] [--preview]
"""

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("txt2ls")

# ---------------------------------------------------------------------------
# VALID LABELS
# ---------------------------------------------------------------------------
VALID_LABELS: set[str] = {
    "BRAND", "MODEL", "PRODUCT_TYPE", "QUANTITY", "ORIGIN", "COMPONENT",
    "MATERIAL", "SIZE", "COMPAT", "ATTRIBUTE", "OCCASION", "EFFECT",
    "COLOR", "STYLE", "TARGET_GROUP",
    "SPEC", "CONNECTIVITY",
    "CAPACITY", "POWER",
    "INGREDIENT", "SKIN_TYPE", "BODY_PART", "VOLUME_WEIGHT", "CONCENTRATION",
}

_CSV_FIELDS = [
    "platform", "industry_group",
    "category_l1", "category_l2", "category_l3",
    "brand", "brand_clean",
]

_ID_LINE_RE = re.compile(r"^\[ID:\s*(.+?)\]\s*(.*)", re.DOTALL)
_XML_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    """NFC chỉ dùng nội bộ để so sánh/tìm offset — KHÔNG dùng để lưu."""
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _xml_autofix(text: str) -> str:
    text = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[\da-fA-F]+);)", "&amp;", text)
    text = re.sub(r"<\s+(\w[\w_]*)\s*>", r"<\1>", text)
    text = re.sub(r"<\s*/\s*(\w[\w_]*)\s*>", r"</\1>", text)
    return text


# ---------------------------------------------------------------------------
# STEP 1: LOAD CSV — raw, không biến đổi
# ---------------------------------------------------------------------------

def load_csv_meta(csv_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("id", "")
            if rid:
                meta[rid] = dict(row)
    logger.info(f"Loaded {len(meta)} rows từ {csv_path.name}")
    return meta


# ---------------------------------------------------------------------------
# STEP 2: PARSE TXT — multi-line block, join bằng "\n" để giữ nguyên
# ---------------------------------------------------------------------------

def parse_txt(txt_path: Path) -> list[tuple[str, str]]:
    """
    Parse TXT → list of (id, raw_tagged_content).
    Multi-line: join bằng "\\n" để không làm thay đổi plain text so với CSV.
    """
    records: list[tuple[str, str]] = []
    current_id: Optional[str] = None
    current_lines: list[str] = []

    def flush():
        if current_id is not None:
            content = "\n".join(current_lines)
            records.append((current_id, content))

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            m = _ID_LINE_RE.match(line)
            if m:
                flush()
                current_id = m.group(1).strip()  # ID luôn strip để match CSV key
                current_lines = []
                inline = m.group(2)              # giữ nguyên, không strip
                if inline:
                    current_lines.append(inline)
            elif current_id is not None:
                if line != "":                   # bỏ dòng rỗng, giữ nguyên nội dung
                    current_lines.append(line)
    flush()

    logger.info(f"Parsed {len(records)} records từ {txt_path.name}")
    return records


# ---------------------------------------------------------------------------
# STEP 3: PARSE XML TAGS → [(label, entity_text), ...]
# ---------------------------------------------------------------------------

def parse_tags(tagged: str) -> tuple[list[tuple[str, str]], list[str], bool]:
    warnings: list[str] = []
    entities: list[tuple[str, str]] = []

    if not tagged.strip():
        return entities, ["[XML] tagged content rỗng."], True

    try:
        root = ET.fromstring(f"<root>{tagged}</root>")
    except ET.ParseError as e1:
        fixed = _xml_autofix(tagged)
        try:
            root = ET.fromstring(f"<root>{fixed}</root>")
            warnings.append(f"[XML] Auto-fix applied: {e1}")
        except ET.ParseError as e2:
            return entities, [f"[XML] Parse failed: {e2}"], True

    for child in root:
        label = child.tag.upper().strip()
        text = "".join(child.itertext())
        # Không strip/nfc — giữ raw, chỉ loại bỏ leading/trailing whitespace
        # (whitespace thừa ở rìa tag là lỗi của LLM, không phải text gốc)
        text = text.strip()
        if not text:
            warnings.append(f"[XML] <{label}> rỗng, bỏ qua.")
            continue
        if label not in VALID_LABELS:
            warnings.append(f"[LABEL] '{label}' không hợp lệ → skip. text='{text[:40]}'")
            continue
        entities.append((label, text))

    return entities, warnings, False


def get_plain(tagged: str, preserve_ws: bool = False) -> str:
    """Strip XML tags → plain text từ LLM output.
    preserve_ws=True: giữ nguyên whitespace (dùng khi --strict-copy để so sánh exact).
    preserve_ws=False: collapse whitespace + strip (default, để so sánh loose).
    """
    plain = _XML_TAG_RE.sub("", tagged)
    if preserve_ws:
        return plain
    return re.sub(r"\s+", " ", plain).strip()


# ---------------------------------------------------------------------------
# STEP 4: VERIFY COPY
# ---------------------------------------------------------------------------

def verify_copy(csv_text: str, llm_plain: str, strict: bool = False) -> tuple[bool, str]:
    """
    strict=True : exact match (csv_text == llm_plain)
    strict=False: normalize whitespace trước khi so sánh (default)
    """
    if strict:
        ok = csv_text == llm_plain
    else:
        ok = normalize_ws(csv_text) == normalize_ws(llm_plain)

    if ok:
        return True, ""
    return False, (
        f"LLM đã sửa text (strict={strict}). "
        f"CSV='{csv_text[:60]}' | LLM='{llm_plain[:60]}'"
    )


# ---------------------------------------------------------------------------
# STEP 5: MAP OFFSETS trên csv_text
# ---------------------------------------------------------------------------

def _find_in(haystack: str, needle: str, from_pos: int) -> int:
    """Tìm needle trong haystack từ from_pos.
    Thử raw trước, nếu không được thử NFC (chỉ để tìm vị trí)."""
    pos = haystack.find(needle, from_pos)
    if pos == -1:
        # Thử NFC — chỉ dùng để tìm, không lưu NFC vào output
        h = _nfc(haystack)
        n = _nfc(needle)
        pos = h.find(n, from_pos)
    return pos


def check_overlaps(result_items: list[dict]) -> list[str]:
    """Phát hiện overlapping span sau map. Trả về danh sách warning."""
    warns: list[str] = []
    sorted_items = sorted(result_items, key=lambda r: r["value"]["start"])
    for i in range(len(sorted_items) - 1):
        a = sorted_items[i]["value"]
        b = sorted_items[i + 1]["value"]
        if b["start"] < a["end"]:
            warns.append(
                f"[OVERLAP] [{a['start']}:{a['end']}]{a['labels']} ∩ "
                f"[{b['start']}:{b['end']}]{b['labels']}"
            )
    return warns


def map_offsets(
    csv_text: str,
    entities: list[tuple[str, str]],
    sample_id: str,
) -> tuple[list[dict], list[str], bool]:
    """
    Map entities → char offsets trên csv_text (raw).
    Trả về (result_items, warnings, has_overlap).
    """
    result_items: list[dict] = []
    warnings: list[str] = []
    search_from = 0
    counter = 0

    for label, entity_text in entities:
        start = _find_in(csv_text, entity_text, search_from)

        if start == -1:
            # Thử từ đầu (LLM đôi khi đổi thứ tự)
            start_alt = _find_in(csv_text, entity_text, 0)
            if start_alt != -1:
                warnings.append(
                    f"[OFFSET_RESET] label={label} '{entity_text[:30]}' "
                    f"found at {start_alt} (reset từ {search_from})."
                )
                start = start_alt
            else:
                warnings.append(
                    f"[OFFSET_MISS] label={label} '{entity_text[:40]}' "
                    f"không tìm được trong csv_text."
                )
                continue

        end = start + len(entity_text)

        # Validate: csv_text[start:end] phải khớp entity_text
        extracted = csv_text[start:end]
        if extracted != entity_text:
            # Thử với NFC
            if _nfc(extracted) != _nfc(entity_text):
                warnings.append(
                    f"[VALIDATE] csv_text[{start}:{end}]='{extracted}' "
                    f"≠ '{entity_text}' label={label}"
                )
                continue

        counter += 1
        result_items.append({
            "id": f"result_{counter}",
            "type": "labels",
            "from_name": "ner",
            "to_name": "text",
            "value": {
                "start": start,
                "end": end,
                "text": extracted,   # lấy từ csv_text — source of truth
                "labels": [label],
            },
        })
        search_from = end

    # Check overlap
    overlap_warns = check_overlaps(result_items)
    warnings.extend(overlap_warns)
    has_overlap = len(overlap_warns) > 0

    return result_items, warnings, has_overlap


# ---------------------------------------------------------------------------
# STEP 6: BUILD TASK
# ---------------------------------------------------------------------------

def build_task(
    sample_id: str,
    tagged: str,
    csv_meta: dict[str, dict],
    model_version: str,
    strict_copy: bool = False,
) -> tuple[dict, list[str]]:
    """CSV là source of truth. data.text = csv_text raw."""
    all_warnings: list[str] = []
    need_adjudication = False

    # --- Lookup CSV ---
    row = csv_meta.get(sample_id, {})
    if not row:
        all_warnings.append(f"[META_MISS] id='{sample_id}' không có trong CSV.")
        return {}, all_warnings

    # csv_text: raw, không NFC, không strip
    csv_text = row.get("title_clean", "")
    if not csv_text:
        csv_text = row.get("title", "")
    if not csv_text:
        all_warnings.append(f"[TEXT_EMPTY] id='{sample_id}' csv_text rỗng.")
        return {}, all_warnings

    # --- Verify copy ---
    llm_plain = get_plain(tagged, preserve_ws=strict_copy)
    copy_ok, copy_msg = verify_copy(csv_text, llm_plain, strict=strict_copy)
    if not copy_ok:
        need_adjudication = True
        all_warnings.append(f"[COPY_ERROR] {copy_msg}")

    # --- Parse tags ---
    entities, parse_warns, xml_err = parse_tags(tagged)
    all_warnings.extend(parse_warns)
    if xml_err:
        all_warnings.append(f"[XML_ERROR] predictions rỗng.")

    # --- Map offsets trên csv_text ---
    result_items: list[dict] = []
    has_overlap = False
    if not xml_err and entities:
        result_items, offset_warns, has_overlap = map_offsets(csv_text, entities, sample_id)
        all_warnings.extend(offset_warns)
        if has_overlap:
            need_adjudication = True

    # --- Build data block ---
    data_block: dict = {"id": sample_id, "text": csv_text}

    for field in _CSV_FIELDS:
        val = row.get(field, "")
        if val:
            data_block[field] = val

    # original_text = title gốc chưa clean (để so sánh nếu cần)
    original = row.get("title", "")
    if original and original != csv_text:
        data_block["original_text"] = original

    data_block.update({
        "need_adjudication": need_adjudication,
        "review_status": "pre_annotated",
        "llm_version": model_version,
    })

    task = {
        "data": data_block,
        "predictions": [
            {"model_version": model_version, "result": result_items}
        ],
    }
    return task, all_warnings


# ---------------------------------------------------------------------------
# STEP 7: SANITY CHECK
# ---------------------------------------------------------------------------

def sanity_check(tasks: list[dict]) -> int:
    errors = 0
    for task in tasks:
        text = task["data"]["text"]
        sid = task["data"]["id"]
        for r in task["predictions"][0]["result"]:
            v = r["value"]
            s, e, t = v["start"], v["end"], v["text"]
            if text[s:e] != t:
                logger.error(f"[SANITY] id={sid} text[{s}:{e}]='{text[s:e]}' ≠ '{t}'")
                errors += 1
    if errors == 0:
        logger.info("✅ Sanity check PASSED")
    else:
        logger.error(f"❌ Sanity check FAILED — {errors} hard errors")
    return errors


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert LLM TXT NER output → Label Studio JSON (predictions)."
    )
    p.add_argument("--input",  "-i", required=True,  type=Path)
    p.add_argument("--csv",    "-c", required=True,  type=Path)
    p.add_argument("--output", "-o", required=True,  type=Path)
    p.add_argument("--model-version", default="claude-sonnet-4-20250514")
    p.add_argument("--warn-file",     default=None, type=Path)
    p.add_argument("--strict-copy",   action="store_true",
                   help="Exact copy check (không collapse whitespace)")
    p.add_argument("--preview",       action="store_true",
                   help="In task đầu tiên ra terminal để verify")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    csv_meta = load_csv_meta(args.csv)
    records = parse_txt(args.input)

    tasks: list[dict] = []
    all_warn_rows: list[dict] = []
    stats = {
        "total": len(records), "ok": 0,
        "meta_miss": 0, "copy_error": 0, "xml_error": 0,
        "offset_miss": 0, "overlap": 0,
        "total_spans": 0, "no_spans": 0,
    }

    for idx, (sample_id, tagged) in enumerate(records):
        task, warnings = build_task(
            sample_id, tagged, csv_meta,
            args.model_version, args.strict_copy,
        )

        if not task:
            stats["meta_miss"] += 1
            for w in warnings:
                all_warn_rows.append({"id": sample_id, "warning": w})
            continue

        for w in warnings:
            all_warn_rows.append({"id": sample_id, "warning": w})
            if "[COPY_ERROR]"   in w: stats["copy_error"]  += 1
            if "[XML_ERROR]"    in w: stats["xml_error"]   += 1
            if "[OFFSET_MISS]"  in w: stats["offset_miss"] += 1
            if "[OVERLAP]"      in w: stats["overlap"]     += 1

        n_spans = len(task["predictions"][0]["result"])
        stats["total_spans"] += n_spans
        if n_spans == 0:
            stats["no_spans"] += 1
        if not warnings:
            stats["ok"] += 1

        tasks.append(task)

        if args.verbose and warnings:
            for w in warnings:
                logger.debug(f"  [{sample_id}] {w}")

        if args.preview and idx == 0:
            print("\n" + "=" * 60)
            print(f"PREVIEW: {sample_id}")
            print("=" * 60)
            print(json.dumps(task, ensure_ascii=False, indent=2))
            print("=" * 60 + "\n")

    sanity_check(tasks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    logger.info(f"Output: {len(tasks)} tasks → {args.output}")

    if args.warn_file and all_warn_rows:
        with open(args.warn_file, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "warning"])
            w.writeheader()
            w.writerows(all_warn_rows)
        logger.info(f"Warnings: {len(all_warn_rows)} rows → {args.warn_file}")

    logger.info("--- SUMMARY ---")
    logger.info(f"total:        {stats['total']}")
    logger.info(f"ok:           {stats['ok']}")
    logger.info(f"meta_miss:    {stats['meta_miss']}")
    logger.info(f"copy_error:   {stats['copy_error']}  (need_adjudication=True)")
    logger.info(f"xml_error:    {stats['xml_error']}")
    logger.info(f"offset_miss:  {stats['offset_miss']} spans")
    logger.info(f"overlap:      {stats['overlap']} spans")
    logger.info(f"total_spans:  {stats['total_spans']}")
    logger.info(f"no_spans:     {stats['no_spans']} tasks")


if __name__ == "__main__":
    main()
