"""
calculate_iaa.py  —  Inter-Annotator Agreement cho Span-based NER
==================================================================
Tính IAA giữa 2 annotators dựa trên character-offset spans.

USAGE:
    python calculate_iaa.py --a annotator_A.jsonl --b annotator_B.jsonl
    python calculate_iaa.py --a annotator_A.jsonl --b annotator_B.jsonl \\
        --outdir iaa_output

INPUT FORMAT (mỗi dòng 1 JSON):
    {
        "id": "sample_001",
        "text": "Áo thun Nike size L",
        "entities": [
            {"start_char": 0, "end_char": 7,  "label": "PRODUCT_TYPE", "text": "Áo thun"},
            {"start_char": 8, "end_char": 12, "label": "BRAND",        "text": "Nike"}
        ]
    }

OUTPUT:
    iaa_report.json          — tất cả metrics dạng JSON
    per_label_f1.csv         — exact F1 từng label
    label_confusion.csv      — confusion matrix (boundary-matched pairs)
    disagreements.csv        — chi tiết từng conflict
    validation_warnings.csv  — span validate failures
"""

import argparse
import csv
import json
import logging
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# RULE HINTS — Mapping (conflict_type, label_a, label_b) → gợi ý Guideline
# Dựa trên R01–R15 và Golden Rules từ Annotation Guideline v3.0
# ---------------------------------------------------------------------------

RULE_HINTS: dict[tuple[str, str, str], str] = {
    # R01: PRODUCT_TYPE vs ATTRIBUTE
    ("label_error", "PRODUCT_TYPE", "ATTRIBUTE"): "R01 — Drop-test: bỏ modifier đi còn hiểu loại SP không? Có→PRODUCT_TYPE, Không→ATTRIBUTE",
    ("label_error", "ATTRIBUTE", "PRODUCT_TYPE"): "R01 — Drop-test: bỏ modifier đi còn hiểu loại SP không? Có→PRODUCT_TYPE, Không→ATTRIBUTE",
    # R02: PRODUCT_TYPE vs CONNECTIVITY
    ("label_error", "PRODUCT_TYPE", "CONNECTIVITY"): "R02 — Cụm từ kết nối đã thành tên gọi phổ thông? Có→PRODUCT_TYPE, Không→CONNECTIVITY",
    ("label_error", "CONNECTIVITY", "PRODUCT_TYPE"): "R02 — Cụm từ kết nối đã thành tên gọi phổ thông? Có→PRODUCT_TYPE, Không→CONNECTIVITY",
    # R03: PRODUCT_TYPE vs MATERIAL
    ("label_error", "PRODUCT_TYPE", "MATERIAL"): "R03 — Tên chất liệu tạo thành tên gọi phổ thông? Có (Nồi Inox)→PRODUCT_TYPE, Không→MATERIAL",
    ("label_error", "MATERIAL", "PRODUCT_TYPE"): "R03 — Tên chất liệu tạo thành tên gọi phổ thông? Có (Nồi Inox)→PRODUCT_TYPE, Không→MATERIAL",
    # R04: ATTRIBUTE vs STYLE
    ("label_error", "ATTRIBUTE", "STYLE"): "R04 — Form dáng/hoa văn vật lý→STYLE. Tính năng kỹ thuật đo được→ATTRIBUTE",
    ("label_error", "STYLE", "ATTRIBUTE"): "R04 — Form dáng/hoa văn vật lý→STYLE. Tính năng kỹ thuật đo được→ATTRIBUTE",
    # R05: EFFECT vs OCCASION
    ("label_error", "EFFECT", "OCCASION"): "R05 — Mô tả mục đích/hoàn cảnh/mùa vụ→OCCASION. Lợi ích sinh học nhận được→EFFECT",
    ("label_error", "OCCASION", "EFFECT"): "R05 — Mô tả mục đích/hoàn cảnh/mùa vụ→OCCASION. Lợi ích sinh học nhận được→EFFECT",
    # R06: EFFECT vs ATTRIBUTE
    ("label_error", "EFFECT", "ATTRIBUTE"): "R06 — Domain Beauty & Health + lợi ích cơ thể→EFFECT. Ngành khác→ATTRIBUTE",
    ("label_error", "ATTRIBUTE", "EFFECT"): "R06 — Domain Beauty & Health + lợi ích cơ thể→EFFECT. Ngành khác→ATTRIBUTE",
    # R07: SPEC vs POWER
    ("label_error", "SPEC", "POWER"): "R07 — Cổng sạc/truyền dữ liệu (W đầu ra)→SPEC. Gia dụng tiêu thụ điện→POWER",
    ("label_error", "POWER", "SPEC"): "R07 — Cổng sạc/truyền dữ liệu (W đầu ra)→SPEC. Gia dụng tiêu thụ điện→POWER",
    # R08: SIZE vs COMPAT
    ("label_error", "SIZE", "COMPAT"): "R08 — Kích thước của chính SP bán→SIZE. Kích thước thiết bị được phục vụ→COMPAT",
    ("label_error", "COMPAT", "SIZE"): "R08 — Kích thước của chính SP bán→SIZE. Kích thước thiết bị được phục vụ→COMPAT",
    # R09: COMPONENT vs PRODUCT_TYPE
    ("label_error", "COMPONENT", "PRODUCT_TYPE"): "R09 — Có anchor word (kèm/tặng/+)? Có→phần sau là COMPONENT. Không→PRODUCT_TYPE",
    ("label_error", "PRODUCT_TYPE", "COMPONENT"): "R09 — Có anchor word (kèm/tặng/+)? Có→phần sau là COMPONENT. Không→PRODUCT_TYPE",
    # R10: MODEL vs BRAND
    ("label_error", "MODEL", "BRAND"): "R10 — Tên công ty/tập đoàn→BRAND. Mã định danh định vị sản phẩm→MODEL",
    ("label_error", "BRAND", "MODEL"): "R10 — Tên công ty/tập đoàn→BRAND. Mã định danh định vị sản phẩm→MODEL",
    # R11: MODEL vs STYLE
    ("label_error", "MODEL", "STYLE"): "R11 — Mã kỹ thuật alphanumeric→MODEL. Tên mẫu/hoa văn thời trang→STYLE",
    ("label_error", "STYLE", "MODEL"): "R11 — Mã kỹ thuật alphanumeric→MODEL. Tên mẫu/hoa văn thời trang→STYLE",
    # R12: ATTRIBUTE vs OCCASION (Home)
    ("label_error", "ATTRIBUTE", "OCCASION"): "R12 — Chỉ phòng/không gian lắp đặt hoặc mục đích gia dụng→OCCASION. Không→ATTRIBUTE",
    ("label_error", "OCCASION", "ATTRIBUTE"): "R12 — Chỉ phòng/không gian lắp đặt hoặc mục đích gia dụng→OCCASION. Không→ATTRIBUTE",
    # R13: CAPACITY vs VOLUME_WEIGHT
    ("label_error", "CAPACITY", "VOLUME_WEIGHT"): "R13 — Domain Beauty ml/g mỹ phẩm→VOLUME_WEIGHT. Home L/kg tải trọng gia dụng→CAPACITY",
    ("label_error", "VOLUME_WEIGHT", "CAPACITY"): "R13 — Domain Beauty ml/g mỹ phẩm→VOLUME_WEIGHT. Home L/kg tải trọng gia dụng→CAPACITY",
    # R14: SPEC vs QUANTITY
    ("label_error", "SPEC", "QUANTITY"): "R14 — Năng lực kỹ thuật→SPEC. Số lượng đơn vị giao dịch→QUANTITY",
    ("label_error", "QUANTITY", "SPEC"): "R14 — Năng lực kỹ thuật→SPEC. Số lượng đơn vị giao dịch→QUANTITY",
    # R15: OCCASION vs TARGET_GROUP
    ("label_error", "OCCASION", "TARGET_GROUP"): "R15 — Chỉ đối tượng người dùng→TARGET_GROUP. Chỉ hoàn cảnh/dịp→OCCASION",
    ("label_error", "TARGET_GROUP", "OCCASION"): "R15 — Chỉ đối tượng người dùng→TARGET_GROUP. Chỉ hoàn cảnh/dịp→OCCASION",
}

# Hint cho boundary_error (cùng label nhưng biên lệch)
_BOUNDARY_RULE_HINTS: dict[str, str] = {
    "MODEL":         "§Boundary MODEL: Bao toàn chuỗi kể cả dấu gạch ngang, mã vùng (EU/VN/A), version suffix (Pro Max)",
    "COMPAT":        "§5.1 COMPAT liên tiếp BẮT BUỘC TÁCH từng thiết bị riêng. Dấu / giữa thiết bị → O",
    "CONNECTIVITY":  "§5.1 CONNECTIVITY liên tiếp BẮT BUỘC TÁCH từng chuẩn riêng. Dấu / giữa → O",
    "SIZE":          "§5.2 Số + đơn vị kích thước LUÔN đi cùng 1 span (VD: [27 inch]_SIZE). Không tách số và đơn vị",
    "SPEC":          "§5.2 Số + đơn vị kỹ thuật LUÔN đi cùng (VD: [50000mAh]_SPEC). Không tách số và đơn vị",
    "CAPACITY":      "§5.2 Số + đơn vị tải trọng LUÔN đi cùng (VD: [7.5 kg]_CAPACITY)",
    "VOLUME_WEIGHT": "§5.2 Số + đơn vị dung tích mỹ phẩm LUÔN đi cùng (VD: [100ml]_VOLUME_WEIGHT)",
    "CONCENTRATION": "§5.2 Số + ký tự đặc biệt/% LUÔN đi cùng (VD: [SPF50+]_CONCENTRATION, [10%]_CONCENTRATION)",
    "QUANTITY":      "G02: Giới từ (cho/dành cho/kèm) KHÔNG bao vào span. Span QUANTITY = số + đơn vị đếm",
    "BRAND":         "§Boundary BRAND: Span chỉ bao tên thương hiệu thuần. Không kéo sang tên dòng SP hay MODEL",
    "PRODUCT_TYPE":  "G08: PRODUCT_TYPE xuất hiện nhiều lần → chỉ gán lần đầu đầy đủ nhất. Lần sau → O",
}


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("iaa")


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """Một entity span đã được validate và normalize."""
    start: int
    end: int
    label: str
    text: str          # text[start:end] sau normalize — để audit


@dataclass
class Sample:
    """Một sample đã load và validate."""
    sample_id: str
    text: str          # NFC-normalized
    spans: list[Span] = field(default_factory=list)


@dataclass
class ValidationWarning:
    """Ghi lại span bị drop do validate fail."""
    sample_id: str
    annotator: str     # "A" hoặc "B"
    raw_start: int
    raw_end: int
    raw_label: str
    raw_text: str
    reason: str


@dataclass
class DisagreementRow:
    """Một dòng trong disagreements.csv."""
    sample_id: str
    text: str          # title (truncated)
    conflict_type: str
    a_start: str
    a_end: str
    a_label: str
    a_text: str
    b_start: str
    b_end: str
    b_label: str
    b_text: str
    note: str
    rule_hint: str = ""  # gợi ý luật Guideline bị vi phạm (tự động từ RULE_HINTS)


# ---------------------------------------------------------------------------
# STEP 1: LOAD VÀ NORMALIZE
# ---------------------------------------------------------------------------

def nfc(s: str) -> str:
    """NFC-normalize chuỗi."""
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""


def strip_span_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Chỉnh lại (start, end) để loại bỏ whitespace đầu/cuối của span.
    Ví dụ: text=" Nike ", start=0,end=6 → start=1,end=5 ("Nike").
    Không thay đổi nếu span đã sạch.
    """
    span_text = text[start:end]
    stripped = span_text.lstrip()
    start += len(span_text) - len(stripped)
    stripped = stripped.rstrip()
    end = start + len(stripped)
    return start, end


def validate_and_normalize_span(
    sample_id: str,
    annotator: str,
    text: str,
    raw: dict,
    warnings: list[ValidationWarning],
) -> Optional[Span]:
    """
    Validate một span dict thô:
    1. Parse start_char, end_char, label, text.
    2. NFC normalize entity text.
    3. Strip whitespace biên span (điều chỉnh start/end).
    4. Validate: 0 <= start < end <= len(text) và text[start:end] == entity.text.

    Trả về Span nếu hợp lệ, None nếu không.
    """
    raw_start = raw.get("start_char", -1)
    raw_end   = raw.get("end_char",   -1)
    raw_label = str(raw.get("label", "")).strip().upper()
    raw_text  = nfc(str(raw.get("text", "")))

    def drop(reason: str) -> None:
        detailed_reason = f"[{raw_start}:{raw_end}] {reason}"
        warnings.append(ValidationWarning(
            sample_id=sample_id, annotator=annotator,
            raw_start=raw_start, raw_end=raw_end,
            raw_label=raw_label, raw_text=raw_text,
            reason=detailed_reason,
        ))

    # Kiểm tra kiểu dữ liệu cơ bản
    if not isinstance(raw_start, int) or not isinstance(raw_end, int):
        drop("start_char/end_char không phải integer")
        return None

    if not raw_label:
        drop("label rỗng")
        return None

    # Strip whitespace biên
    start, end = strip_span_whitespace(text, raw_start, raw_end)

    # Validate bounds
    if not (0 <= start < end <= len(text)):
        drop(f"bounds invalid sau strip: [{start}:{end}] len={len(text)}")
        return None

    # Validate text khớp
    extracted = text[start:end]
    if extracted != raw_text.strip():
        # Thử so sánh với extracted (trường hợp text field bị lệch)
        if extracted != raw_text:
            drop(
                f"text mismatch: text[{start}:{end}]='{extracted}' "
                f"≠ entity.text='{raw_text}'"
            )
            return None

    return Span(start=start, end=end, label=raw_label, text=extracted)


def load_jsonl(
    path: Path,
    annotator: str,
    warnings: list[ValidationWarning],
    logger: logging.Logger,
) -> dict[str, Sample]:
    """
    Load file JSONL, normalize, validate từng span.
    Trả về dict {sample_id → Sample}.
    """
    if not path.exists():
        logger.error(f"File không tồn tại: {path}")
        sys.exit(1)

    samples: dict[str, Sample] = {}
    dropped_spans = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[{annotator}] Line {lineno}: JSON invalid, skipped. {e}")
                continue

            sid  = str(raw.get("id", f"line_{lineno}"))
            text = nfc(str(raw.get("text", "")))
            raw_entities = raw.get("entities", [])

            valid_spans: list[Span] = []
            for ent in raw_entities:
                span = validate_and_normalize_span(sid, annotator, text, ent, warnings)
                if span is not None:
                    valid_spans.append(span)
                else:
                    dropped_spans += 1

            if sid in samples:
                logger.warning(f"[{annotator}] Duplicate id='{sid}', ghi đè.")
            samples[sid] = Sample(sample_id=sid, text=text, spans=valid_spans)

    logger.info(
        f"[{annotator}] Loaded {len(samples)} samples, "
        f"dropped {dropped_spans} invalid spans."
    )
    return samples


def check_intra_annotator_overlaps(
    samples: dict[str, Sample],
    annotator: str,
    warnings: list[ValidationWarning],
    logger: logging.Logger,
) -> int:
    """
    Kiểm tra overlapping spans trong cùng một annotator (intra-annotator).
    Nếu 2 spans trong cùng sample có overlap > 0 → ghi warning.
    KHÔNG sửa spans, KHÔNG drop spans, chỉ flag.

    Returns:
        Số cặp overlap phát hiện được.
    """
    total_overlap_count = 0

    for sid, sample in samples.items():
        spans = sample.spans
        # So sánh từng cặp (i, j) với i < j để tránh đếm 2 lần
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = spans[i], spans[j]
                ov = max(0, min(a.end, b.end) - max(a.start, b.start))
                if ov > 0:
                    total_overlap_count += 1
                    msg = (
                        f"[{annotator}] sample='{sid}' "
                        f"span1=[{a.start}:{a.end}]'{a.text}'({a.label}) "
                        f"span2=[{b.start}:{b.end}]'{b.text}'({b.label}) "
                        f"overlap={ov} chars"
                    )
                    logger.warning(msg)
                    warnings.append(ValidationWarning(
                        sample_id=sid,
                        annotator=annotator,
                        raw_start=a.start,
                        raw_end=a.end,
                        raw_label=a.label,
                        raw_text=a.text,
                        reason=(
                            f"overlapping_spans_same_annotator: "
                            f"overlaps with [{b.start}:{b.end}]'{b.text}'({b.label}) "
                            f"by {ov} chars"
                        ),
                    ))

    return total_overlap_count


# ---------------------------------------------------------------------------
# STEP 2: MATCH SAMPLES
# ---------------------------------------------------------------------------

def match_samples(
    samples_a: dict[str, Sample],
    samples_b: dict[str, Sample],
    logger: logging.Logger,
) -> list[tuple[Sample, Sample]]:
    """
    Match samples giữa A và B theo id.
    Sample chỉ có ở một bên → warning, bỏ khỏi metrics.
    Trả về list (sample_a, sample_b) chỉ gồm shared ids.
    """
    ids_a = set(samples_a)
    ids_b = set(samples_b)
    shared = ids_a & ids_b

    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    if only_a:
        logger.warning(f"{len(only_a)} samples chỉ có ở A, bỏ qua: {sorted(only_a)[:5]}{'...' if len(only_a) > 5 else ''}")
    if only_b:
        logger.warning(f"{len(only_b)} samples chỉ có ở B, bỏ qua: {sorted(only_b)[:5]}{'...' if len(only_b) > 5 else ''}")

    logger.info(f"Shared samples: {len(shared)} / A:{len(ids_a)} B:{len(ids_b)}")
    return [(samples_a[sid], samples_b[sid]) for sid in sorted(shared)]


# ---------------------------------------------------------------------------
# STEP 3: METRICS HELPERS
# ---------------------------------------------------------------------------

def overlap(a: Span, b: Span) -> int:
    """Tính số ký tự overlap giữa 2 span."""
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Tính Precision, Recall, F1 từ tp/fp/fn."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _as_key(s: Span) -> tuple[int, int, str]:
    return (s.start, s.end, s.label)

def _as_boundary(s: Span) -> tuple[int, int]:
    return (s.start, s.end)


# ---------------------------------------------------------------------------
# STEP 4: EXACT MATCH F1
# ---------------------------------------------------------------------------

def compute_exact_f1(
    pairs: list[tuple[Sample, Sample]],
) -> tuple[float, float, float, int, int, int]:
    """
    Micro Exact Span F1: một match khi (start, end, label) trùng tuyệt đối.
    Trả về (P, R, F1, tp, fp, fn).
    """
    tp = fp = fn = 0
    for sa, sb in pairs:
        keys_a = {_as_key(s) for s in sa.spans}
        keys_b = {_as_key(s) for s in sb.spans}
        tp += len(keys_a & keys_b)
        fp += len(keys_a - keys_b)   # A có, B không → false positive của A
        fn += len(keys_b - keys_a)   # B có, A không → false negative của A
    p, r, f = prf(tp, fp, fn)
    return p, r, f, tp, fp, fn


# ---------------------------------------------------------------------------
# STEP 5: BOUNDARY-ONLY F1
# ---------------------------------------------------------------------------

def compute_boundary_f1(
    pairs: list[tuple[Sample, Sample]],
) -> tuple[float, float, float]:
    """
    Boundary-only F1: match khi (start, end) trùng, bỏ qua label.
    """
    tp = fp = fn = 0
    for sa, sb in pairs:
        bounds_a = {_as_boundary(s) for s in sa.spans}
        bounds_b = {_as_boundary(s) for s in sb.spans}
        tp += len(bounds_a & bounds_b)
        fp += len(bounds_a - bounds_b)
        fn += len(bounds_b - bounds_a)
    p, r, f = prf(tp, fp, fn)
    return p, r, f


# ---------------------------------------------------------------------------
# STEP 6: PARTIAL MATCH F1
# ---------------------------------------------------------------------------

def compute_partial_f1(
    pairs: list[tuple[Sample, Sample]],
) -> tuple[float, float, float]:
    """
    Partial Match F1: match nếu overlap > 0.
    - Ưu tiên match cùng label trước.
    - Mỗi span chỉ được match tối đa 1 lần (greedy matching).
    """
    total_tp = total_fp = total_fn = 0

    for sa, sb in pairs:
        matched_a: set[int] = set()  # index trong sa.spans
        matched_b: set[int] = set()  # index trong sb.spans

        # Pass 1: match cùng label + overlap > 0
        for i, a in enumerate(sa.spans):
            for j, b in enumerate(sb.spans):
                if j in matched_b:
                    continue
                if a.label == b.label and overlap(a, b) > 0:
                    matched_a.add(i)
                    matched_b.add(j)
                    break

        # Pass 2: match bất kỳ label + overlap > 0 (cho các span chưa match)
        for i, a in enumerate(sa.spans):
            if i in matched_a:
                continue
            for j, b in enumerate(sb.spans):
                if j in matched_b:
                    continue
                if overlap(a, b) > 0:
                    matched_a.add(i)
                    matched_b.add(j)
                    break

        tp = len(matched_a)
        fp = len(sa.spans) - tp
        fn = len(sb.spans) - len(matched_b)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    p, r, f = prf(total_tp, total_fp, total_fn)
    return p, r, f


# ---------------------------------------------------------------------------
# STEP 7: LABEL AGREEMENT ON MATCHED BOUNDARIES
# ---------------------------------------------------------------------------

def compute_label_agreement(
    pairs: list[tuple[Sample, Sample]],
) -> tuple[float, dict[tuple[str, str], int]]:
    """
    Trong các span có cùng (start, end), tính tỷ lệ label giống nhau.
    Trả về (agreement_rate, confusion_counter).
    confusion_counter: {(label_a, label_b): count}
    """
    total_boundary_matches = 0
    label_agree = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for sa, sb in pairs:
        # Index spans B theo boundary
        b_by_boundary: dict[tuple[int, int], list[Span]] = defaultdict(list)
        for s in sb.spans:
            b_by_boundary[_as_boundary(s)].append(s)

        for a in sa.spans:
            key = _as_boundary(a)
            if key in b_by_boundary:
                # Nếu nhiều span B cùng boundary (hiếm), lấy cái đầu
                b = b_by_boundary[key][0]
                total_boundary_matches += 1
                confusion[(a.label, b.label)] += 1
                if a.label == b.label:
                    label_agree += 1

    rate = label_agree / total_boundary_matches if total_boundary_matches > 0 else 0.0
    return rate, dict(confusion)


# ---------------------------------------------------------------------------
# STEP 8: PER-LABEL EXACT F1
# ---------------------------------------------------------------------------

def compute_per_label_f1(
    pairs: list[tuple[Sample, Sample]],
) -> dict[str, dict[str, float]]:
    """
    Tính exact F1 riêng cho từng label.
    Trả về {label: {"precision": ..., "recall": ..., "f1": ..., "tp":..., "fp":..., "fn":...}}.
    """
    # Gom tp/fp/fn theo label
    label_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sa, sb in pairs:
        keys_a = {_as_key(s) for s in sa.spans}
        keys_b = {_as_key(s) for s in sb.spans}
        for key in keys_a & keys_b:
            label_stats[key[2]]["tp"] += 1
        for key in keys_a - keys_b:
            label_stats[key[2]]["fp"] += 1
        for key in keys_b - keys_a:
            label_stats[key[2]]["fn"] += 1

    result: dict[str, dict[str, float]] = {}
    for label, st in sorted(label_stats.items()):
        p, r, f = prf(st["tp"], st["fp"], st["fn"])
        result[label] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
            "support_a": st["tp"] + st["fp"],  # tổng A có label này
            "support_b": st["tp"] + st["fn"],  # tổng B có label này
        }
    return result


# ---------------------------------------------------------------------------
# STEP 9: COHEN'S KAPPA (CHAR-LEVEL)
# ---------------------------------------------------------------------------

def compute_char_kappa(
    pairs: list[tuple[Sample, Sample]],
    logger: logging.Logger,
) -> tuple[float, float]:
    """
    Tính Cohen's Kappa ở mức character:

    a) WITH O: mỗi char → label entity hoặc "O"
    b) WITHOUT O: chỉ tính trên char có ít nhất 1 annotator gán entity

    Trả về (kappa_with_O, kappa_without_O).

    Lưu ý: Nếu 1 char nằm trong nhiều entity (overlap), raise warning và
    dùng label đầu tiên (theo start).
    """
    labels_a_with_O:    list[str] = []
    labels_b_with_O:    list[str] = []
    labels_a_without_O: list[str] = []
    labels_b_without_O: list[str] = []

    for sa, sb in pairs:
        text_len = len(sa.text)

        # Tạo mảng label theo char cho từng annotator
        # Khởi tạo tất cả = "O"
        char_a = ["O"] * text_len
        char_b = ["O"] * text_len

        # Fill A
        for span in sorted(sa.spans, key=lambda s: s.start):
            for c in range(span.start, span.end):
                if char_a[c] != "O":
                    logger.warning(
                        f"[A] sample='{sa.sample_id}' char={c} overlap: "
                        f"'{char_a[c]}' bị ghi đè bởi '{span.label}'"
                    )
                char_a[c] = span.label

        # Fill B
        for span in sorted(sb.spans, key=lambda s: s.start):
            for c in range(span.start, span.end):
                if char_b[c] != "O":
                    logger.warning(
                        f"[B] sample='{sb.sample_id}' char={c} overlap: "
                        f"'{char_b[c]}' bị ghi đè bởi '{span.label}'"
                    )
                char_b[c] = span.label

        # Gom vào list toàn cục
        labels_a_with_O.extend(char_a)
        labels_b_with_O.extend(char_b)

        # Without O: chỉ lấy char mà ít nhất 1 annotator gán entity
        for a_label, b_label in zip(char_a, char_b):
            if a_label != "O" or b_label != "O":
                labels_a_without_O.append(a_label)
                labels_b_without_O.append(b_label)

    # Tính kappa
    def safe_kappa(y1: list[str], y2: list[str], name: str) -> float:
        if len(set(y1) | set(y2)) < 2:
            logger.warning(f"Kappa {name}: chỉ có 1 class, không thể tính. Trả về 0.0")
            return 0.0
        if len(y1) == 0:
            logger.warning(f"Kappa {name}: không có data. Trả về 0.0")
            return 0.0
        return float(cohen_kappa_score(y1, y2))

    kappa_with_O    = safe_kappa(labels_a_with_O,    labels_b_with_O,    "with_O")
    kappa_without_O = safe_kappa(labels_a_without_O, labels_b_without_O, "without_O")

    return kappa_with_O, kappa_without_O


# ---------------------------------------------------------------------------
# STEP 10: DISAGREEMENT ANALYSIS
# ---------------------------------------------------------------------------

def classify_conflict(
    a: Optional[Span],
    b: Optional[Span],
) -> str:
    """
    Phân loại conflict giữa 2 span (một trong hai có thể là None):
    - exact_match      : (start, end, label) trùng
    - label_error      : (start, end) trùng, label khác
    - boundary_error   : label trùng, overlap > 0, boundary khác
    - partial_overlap  : overlap > 0, cả boundary lẫn label đều khác
    - missing_in_A     : A=None
    - missing_in_B     : B=None
    """
    if a is None:
        return "missing_in_A"
    if b is None:
        return "missing_in_B"
    if a.start == b.start and a.end == b.end and a.label == b.label:
        return "exact_match"
    if a.start == b.start and a.end == b.end:
        return "label_error"
    if a.label == b.label and overlap(a, b) > 0:
        return "boundary_error"
    if overlap(a, b) > 0:
        return "partial_overlap"
    # Không overlap → treat as missing_in_B / missing_in_A
    return "missing_in_B"  # fallback (không nên xảy ra nếu caller đúng)


def get_rule_hint(conflict_type: str, a_label: str, b_label: str) -> str:
    """
    Tra cứu gợi ý luật Guideline dựa trên loại conflict và cặp nhãn.
    Trả về chuỗi mô tả rule cần check, hoặc "" nếu không có mapping.
    """
    key = (conflict_type, a_label, b_label)
    if key in RULE_HINTS:
        return RULE_HINTS[key]
    if conflict_type == "boundary_error":
        return _BOUNDARY_RULE_HINTS.get(a_label, "")
    if conflict_type in ("missing_in_A", "missing_in_B"):
        return "G14 — Khi không chắc → log vào Adjudication Queue. Kiểm tra lại INCLUDE/EXCLUDE trong Guideline"
    return ""


def build_disagreements(
    pairs: list[tuple[Sample, Sample]],
) -> list[DisagreementRow]:
    """
    Tạo danh sách disagreement rows cho disagreements.csv.

    Thuật toán:
    1. Exact matches → ghi với conflict_type="exact_match".
    2. Matched boundaries khác label → label_error.
    3. Cùng label, overlap > 0, boundary khác → boundary_error.
    4. Overlap nhưng cả label lẫn boundary khác → partial_overlap.
    5. Span A không có match → missing_in_B.
    6. Span B không có match → missing_in_A.
    """
    rows: list[DisagreementRow] = []

    def row(
        sid: str,
        text: str,
        ctype: str,
        a: Optional[Span],
        b: Optional[Span],
        note: str = "",
    ) -> DisagreementRow:
        a_label = a.label if a else ""
        b_label = b.label if b else ""
        return DisagreementRow(
            sample_id=sid,
            text=text[:80],
            conflict_type=ctype,
            a_start=str(a.start) if a else "",
            a_end=str(a.end)   if a else "",
            a_label=a_label,
            a_text=a.text      if a else "",
            b_start=str(b.start) if b else "",
            b_end=str(b.end)   if b else "",
            b_label=b_label,
            b_text=b.text      if b else "",
            note=note,
            rule_hint=get_rule_hint(ctype, a_label, b_label),
        )

    for sa, sb in pairs:
        sid  = sa.sample_id
        text = sa.text

        matched_a: set[int] = set()
        matched_b: set[int] = set()

        # Pass 1: exact match (start, end, label)
        exact_keys_a = {_as_key(s): i for i, s in enumerate(sa.spans)}
        for j, b in enumerate(sb.spans):
            key = _as_key(b)
            if key in exact_keys_a:
                i = exact_keys_a[key]
                matched_a.add(i)
                matched_b.add(j)
                rows.append(row(sid, text, "exact_match", sa.spans[i], b))

        # Pass 2: same boundary, different label → label_error
        bound_a = {_as_boundary(s): i for i, s in enumerate(sa.spans)
                   if i not in matched_a}
        for j, b in enumerate(sb.spans):
            if j in matched_b:
                continue
            bkey = _as_boundary(b)
            if bkey in bound_a:
                i = bound_a[bkey]
                matched_a.add(i)
                matched_b.add(j)
                rows.append(row(
                    sid, text, "label_error", sa.spans[i], b,
                    note=f"boundary match, label A={sa.spans[i].label} B={b.label}",
                ))

        # Pass 3: same label + overlap > 0, boundary khác → boundary_error
        remaining_a = [(i, s) for i, s in enumerate(sa.spans) if i not in matched_a]
        remaining_b = [(j, s) for j, s in enumerate(sb.spans) if j not in matched_b]

        for i, a in remaining_a:
            for j, b in remaining_b:
                if j in matched_b:
                    continue
                if a.label == b.label and overlap(a, b) > 0:
                    matched_a.add(i)
                    matched_b.add(j)
                    rows.append(row(
                        sid, text, "boundary_error", a, b,
                        note=f"label={a.label}, overlap={overlap(a,b)}",
                    ))
                    break

        # Pass 4: partial overlap (label khác, boundary khác, overlap > 0)
        remaining_a = [(i, s) for i, s in enumerate(sa.spans) if i not in matched_a]
        remaining_b = [(j, s) for j, s in enumerate(sb.spans) if j not in matched_b]

        for i, a in remaining_a:
            for j, b in remaining_b:
                if j in matched_b:
                    continue
                if overlap(a, b) > 0:
                    matched_a.add(i)
                    matched_b.add(j)
                    rows.append(row(
                        sid, text, "partial_overlap", a, b,
                        note=f"overlap={overlap(a,b)}",
                    ))
                    break

        # Pass 5: unmatched A → missing_in_B
        for i, a in enumerate(sa.spans):
            if i not in matched_a:
                rows.append(row(sid, text, "missing_in_B", a, None))

        # Pass 6: unmatched B → missing_in_A
        for j, b in enumerate(sb.spans):
            if j not in matched_b:
                rows.append(row(sid, text, "missing_in_A", None, b))

    return rows


# ---------------------------------------------------------------------------
# STEP 11: WRITE OUTPUT FILES
# ---------------------------------------------------------------------------

def write_iaa_report(
    outdir: Path,
    shared_count: int,
    total_a: int,
    total_b: int,
    exact_prf: tuple,
    boundary_prf: tuple,
    partial_prf: tuple,
    label_agreement_rate: float,
    kappa_with_O: float,
    kappa_without_O: float,
    confusion: dict,
    per_label: dict,
    intra_overlap_count: int = 0,
    conflict_distribution: Optional[dict] = None,
) -> None:
    """Ghi iaa_report.json."""
    # Top 10 label confusions (sắp xếp theo count, bỏ exact match)
    top_confusions = sorted(
        [{"a_label": k[0], "b_label": k[1], "count": v}
         for k, v in confusion.items() if k[0] != k[1]],
        key=lambda x: -x["count"],
    )[:10]

    glossary = {
        "_note": "Phần này giải thích ý nghĩa các thuật ngữ trong report.",
        "annotator_A": "YenTran (24522070@gm.uit.edu.vn) — người gán nhãn A.",
        "annotator_B": "YenDo — người gán nhãn B.",
        "shared_samples": "Số sample cả 2 annotator cùng gán nhãn (dùng để tính IAA).",
        "total_entities_A_B": "Tổng số entity span mà A (hoặc B) đã gán nhãn trên toàn bộ shared samples.",
        "TP_true_positive": "Span A và span B giống nhau hoàn toàn (start, end, label đều trùng) → cả 2 đồng thuận.",
        "FP_false_positive": "Span A gán nhãn nhưng B không gán (hoặc gán khác) → A 'thừa' so với B.",
        "FN_false_negative": "Span B gán nhãn nhưng A không gán (hoặc gán khác) → A 'thiếu' so với B.",
        "precision": "Trong số span A gán, bao nhiêu % trùng với B. Precision = TP / (TP + FP). Precision thấp → A gán quá nhiều / gán sai.",
        "recall": "Trong số span B gán, A bắt được bao nhiêu %. Recall = TP / (TP + FN). Recall thấp → A bỏ sót nhiều span mà B đã gán.",
        "f1": "Trung bình điều hòa của Precision và Recall. F1 = 2*P*R / (P+R). Metric tổng hợp để so sánh mức độ đồng thuận.",
        "exact_span_f1": "F1 nghiêm ngặt nhất: span coi là khớp khi (start, end, label) đều trùng tuyệt đối.",
        "boundary_only_f1": "F1 chỉ xét biên span (start, end), bỏ qua label. Cho thấy 2 annotator có đồng thuận về VỊ TRÍ entity hay không, tách biệt với vấn đề chọn nhãn.",
        "partial_match_f1": "F1 khi chỉ cần overlap > 0 ký tự là coi là khớp (có ưu tiên cùng label trước). Metric 'dễ tính' nhất, phản ánh mức đồng thuận tổng quát.",
        "label_agreement_on_matched_boundaries": "Trong các span có biên (start, end) trùng nhau, tỷ lệ % 2 annotator chọn cùng label. Đây là metric đo chất lượng hiểu label riêng biệt với biên.",
        "cohen_kappa_with_O": "Cohen's Kappa tính ở mức ký tự, kể cả ký tự không thuộc entity nào (gán nhãn 'O'). Bị inflate cao vì đa số ký tự là 'O' và cả 2 đều đồng ý → metric này không phản ánh thực chất.",
        "cohen_kappa_without_O": "Cohen's Kappa chỉ tính trên ký tự thuộc ít nhất 1 entity. Phản ánh thực chất mức đồng thuận nhãn hơn kappa_with_O. Thang đánh giá: <0.4=yếu, 0.4-0.6=trung bình, 0.6-0.8=tốt, >0.8=rất tốt.",
        "conflict_types": {
            "exact_match": "A và B gán hoàn toàn giống nhau (start, end, label). Đây là trường hợp lý tưởng.",
            "label_error": "A và B có cùng biên span (start, end) nhưng chọn label khác nhau → bất đồng về phân loại.",
            "boundary_error": "A và B chọn cùng label, span có overlap, nhưng biên (start hoặc end) lệch nhau → bất đồng về độ chính xác biên.",
            "partial_overlap": "Span có overlap nhưng cả label lẫn biên đều không khớp.",
            "missing_in_B": "A gán nhãn span này nhưng B không gán → B bỏ sót.",
            "missing_in_A": "B gán nhãn span này nhưng A không gán → A bỏ sót."
        },
        "support_a": "Tổng số span có label này mà A đã gán (TP + FP).",
        "support_b": "Tổng số span có label này mà B đã gán (TP + FN).",
    }

    report = {
        "_glossary": glossary,
        "shared_samples": shared_count,
        "overlapping_span_warnings_count": intra_overlap_count,
        "total_entities_A": total_a,
        "total_entities_B": total_b,
        "exact_span": {
            "precision": round(exact_prf[0], 4),
            "recall":    round(exact_prf[1], 4),
            "f1":        round(exact_prf[2], 4),
            "tp": exact_prf[3], "fp": exact_prf[4], "fn": exact_prf[5],
        },
        "boundary_only": {
            "precision": round(boundary_prf[0], 4),
            "recall":    round(boundary_prf[1], 4),
            "f1":        round(boundary_prf[2], 4),
        },
        "partial_match": {
            "precision": round(partial_prf[0], 4),
            "recall":    round(partial_prf[1], 4),
            "f1":        round(partial_prf[2], 4),
        },
        "label_agreement_on_matched_boundaries": round(label_agreement_rate, 4),
        "cohen_kappa_with_O":    round(kappa_with_O,    4),
        "cohen_kappa_without_O": round(kappa_without_O, 4),
        "kappa_note": (
            "Kappa là metric phụ vì NER span annotation phù hợp hơn với span-level F1. "
            "kappa_with_O bị kéo lên cao bởi số lượng lớn O characters. "
            "kappa_without_O phản ánh thực chất agreement trên entity characters."
        ),
        "top_label_confusions": top_confusions,
        "per_label_f1": per_label,
        "conflict_type_distribution": conflict_distribution or {},
    }

    path = outdir / "iaa_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_per_label_csv(outdir: Path, per_label: dict) -> None:
    """Ghi per_label_f1.csv."""
    path = outdir / "per_label_f1.csv"
    rows = [{"label": label, **stats} for label, stats in per_label.items()]
    df = pd.DataFrame(rows, columns=[
        "label", "precision", "recall", "f1",
        "tp", "fp", "fn", "support_a", "support_b",
    ])
    df.sort_values("f1", ascending=False, inplace=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_confusion_csv(outdir: Path, confusion: dict) -> None:
    """Ghi label_confusion.csv (boundary-matched pairs)."""
    path = outdir / "label_confusion.csv"
    rows = [{"a_label": k[0], "b_label": k[1], "count": v}
            for k, v in sorted(confusion.items(), key=lambda x: -x[1])]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def write_disagreements_csv(outdir: Path, rows: list[DisagreementRow]) -> None:
    """Ghi disagreements.csv."""
    path = outdir / "disagreements.csv"
    fieldnames = [
        "sample_id", "text", "conflict_type",
        "a_start", "a_end", "a_label", "a_text",
        "b_start", "b_end", "b_label", "b_text",
        "note", "rule_hint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def write_validation_warnings_csv(
    outdir: Path,
    warnings: list[ValidationWarning],
) -> None:
    """Ghi validation_warnings.csv."""
    path = outdir / "validation_warnings.csv"
    fieldnames = [
        "sample_id", "annotator", "raw_start", "raw_end",
        "raw_label", "raw_text", "reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for w in warnings:
            writer.writerow(asdict(w))


# ---------------------------------------------------------------------------
# STEP 12: TERMINAL REPORT
# ---------------------------------------------------------------------------

def print_report(
    shared_count: int,
    total_a: int,
    total_b: int,
    exact_prf: tuple,
    boundary_prf: tuple,
    partial_prf: tuple,
    label_agreement_rate: float,
    kappa_with_O: float,
    kappa_without_O: float,
    confusion: dict,
    per_label: dict,
    logger: logging.Logger,
    intra_overlap_total: int = 0,
) -> None:
    """In báo cáo ra terminal."""
    SEP = "=" * 60

    logger.info(SEP)
    logger.info("IAA REPORT — Vietnamese Ecommerce NER")
    logger.info(SEP)
    logger.info(f"Shared samples:             {shared_count}")
    logger.info(f"Intra-annotator overlap warnings: {intra_overlap_total}")
    logger.info(f"Total entities (A / B):     {total_a} / {total_b}")
    logger.info("---")
    logger.info("EXACT SPAN F1  (start, end, label phải trùng)")
    logger.info(f"  Precision:  {exact_prf[0]:.4f}")
    logger.info(f"  Recall:     {exact_prf[1]:.4f}")
    logger.info(f"  F1:         {exact_prf[2]:.4f}")
    logger.info(f"  TP={exact_prf[3]}  FP={exact_prf[4]}  FN={exact_prf[5]}")
    logger.info("---")
    logger.info("BOUNDARY-ONLY F1  (bỏ qua label)")
    logger.info(f"  Precision:  {boundary_prf[0]:.4f}")
    logger.info(f"  Recall:     {boundary_prf[1]:.4f}")
    logger.info(f"  F1:         {boundary_prf[2]:.4f}")
    logger.info("---")
    logger.info("PARTIAL MATCH F1  (overlap > 0, cùng label ưu tiên)")
    logger.info(f"  Precision:  {partial_prf[0]:.4f}")
    logger.info(f"  Recall:     {partial_prf[1]:.4f}")
    logger.info(f"  F1:         {partial_prf[2]:.4f}")
    logger.info("---")
    logger.info(f"LABEL AGREEMENT on matched boundaries: {label_agreement_rate:.4f}")
    logger.info("---")
    logger.info("COHEN'S KAPPA (char-level)  [metric phụ]")
    logger.info(f"  kappa_with_O:    {kappa_with_O:.4f}  (inflate bởi O chars)")
    logger.info(f"  kappa_without_O: {kappa_without_O:.4f}  (trên entity chars only)")
    logger.info("  NOTE: Kappa phụ — span-level F1 phù hợp hơn cho NER IAA.")
    logger.info("---")
    logger.info("TOP LABEL CONFUSIONS (boundary match, label khác):")
    top = sorted([(k, v) for k, v in confusion.items() if k[0] != k[1]],
                 key=lambda x: -x[1])[:8]
    for (la, lb), cnt in top:
        logger.info(f"  A={la:<20} B={lb:<20} count={cnt}")
    logger.info("---")
    logger.info("PER-LABEL EXACT F1:")
    logger.info(f"  {'Label':<25} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    for label, st in sorted(per_label.items(), key=lambda x: -x[1]["f1"]):
        logger.info(
            f"  {label:<25} {st['precision']:>6.3f} {st['recall']:>6.3f} "
            f"{st['f1']:>6.3f} {st['tp']:>5} {st['fp']:>5} {st['fn']:>5}"
        )
    logger.info(SEP)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tính IAA cho span-based NER dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python calculate_iaa.py --a annotator_A.jsonl --b annotator_B.jsonl\n"
            "  python calculate_iaa.py --a annotator_A.jsonl --b annotator_B.jsonl \\\n"
            "      --outdir iaa_output --verbose"
        ),
    )
    p.add_argument("--a",      required=True, type=Path, help="File JSONL của annotator A.")
    p.add_argument("--b",      required=True, type=Path, help="File JSONL của annotator B.")
    p.add_argument("--outdir", default=Path("iaa_output"), type=Path,
                   help="Thư mục output (default: iaa_output).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.verbose)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load & validate
    val_warnings: list[ValidationWarning] = []
    samples_a = load_jsonl(args.a, "A", val_warnings, logger)
    samples_b = load_jsonl(args.b, "B", val_warnings, logger)

    # Kiểm tra overlapping spans trong cùng annotator (quality check)
    overlap_count_a = check_intra_annotator_overlaps(samples_a, "A", val_warnings, logger)
    overlap_count_b = check_intra_annotator_overlaps(samples_b, "B", val_warnings, logger)
    intra_overlap_total = overlap_count_a + overlap_count_b

    if val_warnings:
        logger.warning(f"{len(val_warnings)} validation warning(s) tổng cộng.")
        write_validation_warnings_csv(args.outdir, val_warnings)

    # 2. Match
    pairs = match_samples(samples_a, samples_b, logger)
    if not pairs:
        logger.error("Không có shared sample. Kiểm tra lại id giữa 2 file.")
        return 1

    shared_count = len(pairs)
    total_a = sum(len(s.spans) for s, _ in pairs)
    total_b = sum(len(s.spans) for _, s in pairs)

    # 3. Metrics
    logger.info("Tính exact span F1...")
    exact_prf   = compute_exact_f1(pairs)

    logger.info("Tính boundary-only F1...")
    bound_prf   = compute_boundary_f1(pairs)

    logger.info("Tính partial match F1...")
    part_prf    = compute_partial_f1(pairs)

    logger.info("Tính label agreement...")
    agree_rate, confusion = compute_label_agreement(pairs)

    logger.info("Tính per-label F1...")
    per_label   = compute_per_label_f1(pairs)

    logger.info("Tính Cohen's Kappa (char-level)...")
    kappa_w, kappa_wo = compute_char_kappa(pairs, logger)

    # 4. Disagreement analysis
    logger.info("Phân tích disagreements...")
    disag_rows = build_disagreements(pairs)

    # 5. Conflict type summary
    ctype_counts: dict[str, int] = defaultdict(int)
    for r in disag_rows:
        ctype_counts[r.conflict_type] += 1
    total_conflicts = sum(ctype_counts.values())
    if total_conflicts > 0:
        logger.info("Conflict distribution:")
        for k, v in sorted(ctype_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  - {k}: {v} ({v / total_conflicts * 100:.1f}%)")
    else:
        logger.info("Conflict distribution: Không có conflict nào!")

    # 6. Print terminal report
    print_report(
        shared_count, total_a, total_b,
        exact_prf, bound_prf, part_prf,
        agree_rate, kappa_w, kappa_wo,
        confusion, per_label, logger,
        intra_overlap_total=intra_overlap_total,
    )

    # 7. Write outputs
    write_iaa_report(
        args.outdir, shared_count, total_a, total_b,
        exact_prf, bound_prf, part_prf,
        agree_rate, kappa_w, kappa_wo,
        confusion, per_label,
        intra_overlap_count=intra_overlap_total,
        conflict_distribution=dict(ctype_counts),
    )
    write_per_label_csv(args.outdir, per_label)
    write_confusion_csv(args.outdir, confusion)
    write_disagreements_csv(args.outdir, disag_rows)

    logger.info(f"Output → {args.outdir}/")
    logger.info("  iaa_report.json")
    logger.info("  per_label_f1.csv")
    logger.info("  label_confusion.csv")
    logger.info("  disagreements.csv")
    if val_warnings:
        logger.info("  validation_warnings.csv")
    logger.info("✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
