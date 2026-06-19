#!/usr/bin/env python3
"""
split_blind_double_annotation.py
================================
Chia dữ liệu annotation sang 4 phần (A, B, C, D) cho 4 annotator theo 
quy trình Blind Double Annotation (Tập QC luân chuyển chéo 15% cuối).

Quy trình:
  1. Load dữ liệu từ campaign_patched_all.json
  2. Shuffle dữ liệu bằng random seed 42 để đảm bảo tính tái lập.
  3. Chia dữ liệu thành 4 phần gần bằng nhau (primary split).
  4. Lấy 15% mẫu cuối của mỗi phần làm tập QC.
  5. Luân chuyển tập QC chéo: QC_A -> B, QC_B -> C, QC_C -> D, QC_D -> A.
  6. Trộn ngẫu nhiên từng tệp của annotator để đảm bảo mù hoàn toàn (blind).
  7. Xuất file mapping riêng blind_mapping.csv cho quản lý dự án.
  8. Xuất A.json, B.json, C.json, D.json sạch nhãn phụ.
  9. In thống kê cuối chương trình.
"""

import argparse
import copy
import csv
import json
import logging
import random
import sys
from pathlib import Path

# Cấu hình logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("split_blind")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS CHO ĐỒNG BỘ DỮ LIỆU ĐỘNG (BẢO ĐẢM KHỚP 100% XML CONFIG)
# ---------------------------------------------------------------------------

COMMON_LABELS = [
    "PRODUCT_TYPE", "BRAND", "MODEL", "QUANTITY", "ORIGIN", "COMPONENT",
    "MATERIAL", "SIZE", "COMPAT", "ATTRIBUTE", "OCCASION", "EFFECT",
    "COLOR", "STYLE", "TARGET_GROUP",
]

DOMAIN_LABELS = {
    "electronics": ["SPEC", "CONNECTIVITY"],
    "home_living": ["POWER", "CAPACITY"],
    "fashion": [],
    "beauty_health": ["INGREDIENT", "SKIN_TYPE", "BODY_PART", "VOLUME_WEIGHT", "CONCENTRATION"],
}

LABELS_ORDERED = {
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


def infer_industry_key(industry: str) -> str:
    raw = (industry or "").lower().strip()
    if not raw:
        return None
    if raw in INDUSTRY_MAP:
        return INDUSTRY_MAP[raw]
    for alias, key in INDUSTRY_MAP.items():
        if alias in raw or raw in alias:
            return key
    return None


def build_label_set(data_dict: dict) -> list:
    industry_group = data_dict.get("industry_group", "")
    industry_key = infer_industry_key(industry_group)
    primary = LABELS_ORDERED.get(industry_key or "", ALL_LABELS)

    final_order = []
    for label in primary + ALL_LABELS:
        if label not in final_order:
            final_order.append(label)

    primary_set = set(primary)
    formatted = []
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


def build_hotkey_hint(labels_for_task: list) -> str:
    labels = [x["value"] for x in labels_for_task[:9]]
    return "  ".join(f"{i + 1}:{label}" for i, label in enumerate(labels))


def build_context_header(data_dict: dict) -> str:
    industry = (data_dict.get("industry_group") or "UNKNOWN").upper()
    cat_l1 = data_dict.get("category_l1", "") or ""
    cat_l2 = data_dict.get("category_l2", "") or ""
    cat_l3 = data_dict.get("category_l3", "") or ""
    
    breadcrumb = " › ".join([cat_l1, cat_l2, cat_l3]).strip(" ›")
    header = f"📁 [{industry}]"
    if breadcrumb:
        header += f"  {breadcrumb}"
        
    brand = data_dict.get("brand") or data_dict.get("catalog_brand")
    if brand and brand not in ("no brand", "No Brand", "no_brand"):
        header += f"  |  🏷️ Brand: {brand}"
    return header


def split_data(
    input_json: Path,
    output_dir: Path,
    seed: int = 42,
    qc_ratio: float = 0.15,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Đọc dữ liệu từ {input_json.name}...")
    with open(input_json, encoding="utf-8") as f:
        tasks = json.load(f)

    total_tasks = len(tasks)
    logger.info(f"Tổng số task đọc được: {total_tasks}")

    # Sanity Check 1: Kiểm tra không có ID trùng trong input
    seen_ids = set()
    for t in tasks:
        sid = t["data"]["id"]
        if sid in seen_ids:
            logger.error(f"Sanity Check FAIL: Phát hiện trùng lặp ID: '{sid}' trong file đầu vào!")
            sys.exit(1)
        seen_ids.add(sid)
    logger.info("✓ Sanity Check 1 Passed: Không có ID trùng lặp trong input.")

    # 1. Shuffle dữ liệu với seed cố định
    logger.info(f"Xáo trộn dữ liệu ngẫu nhiên với seed={seed}...")
    random.seed(seed)
    shuffled_tasks = copy.deepcopy(tasks)
    random.shuffle(shuffled_tasks)

    # 2. Chia thành 4 phần chính (A, B, C, D)
    k = 4
    base_size = total_tasks // k
    remainder = total_tasks % k

    primary_splits = {}
    annotators = ["A", "B", "C", "D"]
    start_idx = 0

    for i, annotator in enumerate(annotators):
        size = base_size + (1 if i < remainder else 0)
        primary_splits[annotator] = shuffled_tasks[start_idx : start_idx + size]
        start_idx += size
        logger.info(f"Primary {annotator}: {len(primary_splits[annotator])} tasks")

    # 3. Tạo tập QC (blind overlap 15% ở cuối phần đó)
    qc_splits = {}
    for annotator in annotators:
        split_tasks = primary_splits[annotator]
        qc_size = int(round(len(split_tasks) * qc_ratio))
        # Lấy 15% ở cuối
        qc_splits[annotator] = split_tasks[-qc_size:]
        logger.info(f"QC_{annotator} (15% cuối của {annotator}): {len(qc_splits[annotator])} tasks")

    # 4. Luân chuyển tập QC
    rotation = {
        "A": "D",  # A nhận QC_D
        "B": "A",  # B nhận QC_A
        "C": "B",  # C nhận QC_B
        "D": "C",  # D nhận QC_C
    }

    final_splits = {}
    for annotator in annotators:
        # Nhận phần chính của mình
        final_splits[annotator] = copy.deepcopy(primary_splits[annotator])
        # Nhận thêm QC từ người đi trước
        from_annotator = rotation[annotator]
        final_splits[annotator].extend(copy.deepcopy(qc_splits[from_annotator]))

    # Sanity Check 2: Kiểm tra tổng số task sau chia = total_tasks + duplicated_qc_tasks
    total_duplicated = sum(len(qc_splits[a]) for a in annotators)
    total_final_tasks = sum(len(final_splits[a]) for a in annotators)
    if total_final_tasks != total_tasks + total_duplicated:
        logger.error(
            f"Sanity Check FAIL: Tổng số task sau chia ({total_final_tasks}) "
            f"không khớp với tổng task gốc + QC ({total_tasks + total_duplicated})!"
        )
        sys.exit(1)
    logger.info("✓ Sanity Check 2 Passed: Tổng số task cuối khớp hoàn hảo (Total + QC).")

    # Sanity Check 3 & 4: Kiểm tra số lần xuất hiện của task thường (1 lần) và task QC (đúng 2 lần)
    from collections import Counter
    all_assigned_ids = []
    for a in annotators:
        all_assigned_ids.extend([t["data"]["id"] for t in final_splits[a]])
    id_counts = Counter(all_assigned_ids)

    qc_ids = set()
    for a in annotators:
        for t in qc_splits[a]:
            qc_ids.add(t["data"]["id"])

    for t in tasks:
        tid = t["data"]["id"]
        actual_count = id_counts[tid]
        if tid in qc_ids:
            if actual_count != 2:
                logger.error(f"Sanity Check FAIL: Task QC {tid} phải xuất hiện đúng 2 lần, thực tế xuất hiện {actual_count} lần!")
                sys.exit(1)
        else:
            if actual_count != 1:
                logger.error(f"Sanity Check FAIL: Task thường {tid} phải xuất hiện đúng 1 lần, thực tế xuất hiện {actual_count} lần!")
                sys.exit(1)
    logger.info("✓ Sanity Check 3 & 4 Passed: Các mẫu thường xuất hiện đúng 1 lần, các mẫu QC xuất hiện đúng 2 lần.")

    # 5. Tạo mapping dữ liệu cho quản lý dự án (blind_mapping.csv)
    mapping_rows = []
    qc_by_task_id = {}
    for from_ann, to_ann in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]:
        for t in qc_splits[from_ann]:
            qc_by_task_id[t["data"]["id"]] = to_ann

    for annotator in annotators:
        for t in primary_splits[annotator]:
            tid = t["data"]["id"]
            sec_ann = qc_by_task_id.get(tid, "")
            is_qc = 1 if sec_ann else 0
            mapping_rows.append({
                "id": tid,
                "primary_annotator": annotator,
                "secondary_annotator": sec_ann,
                "is_qc_sample": is_qc,
                "source_split": annotator
            })

    # Sanity Check 5: Kiểm tra blind_mapping.csv có đúng total_tasks dòng, mỗi ID đúng 1 dòng
    if len(mapping_rows) != total_tasks:
        logger.error(f"Sanity Check FAIL: blind_mapping.csv có {len(mapping_rows)} dòng, không khớp với total_tasks={total_tasks}!")
        sys.exit(1)
    seen_map_ids = set()
    for row in mapping_rows:
        mid = row["id"]
        if mid in seen_map_ids:
            logger.error(f"Sanity Check FAIL: ID {mid} bị lặp lại trong blind_mapping.csv!")
            sys.exit(1)
        seen_map_ids.add(mid)
    logger.info("✓ Sanity Check 5 Passed: blind_mapping.csv chứa đúng số lượng dòng và không có ID trùng lặp.")

    # Ghi file mapping CSV
    mapping_csv_path = output_dir / "blind_mapping.csv"
    with open(mapping_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "primary_annotator", "secondary_annotator", "is_qc_sample", "source_split"
        ])
        writer.writeheader()
        writer.writerows(mapping_rows)
    logger.info(f"Đã xuất file mapping: {mapping_csv_path.name}")

    # 6. Trộn ngẫu nhiên từng tệp JSON của Annotator, chuyển sang định dạng Annotation, đồng bộ from_name và sinh nhãn động
    # Việc trộn ngẫu nhiên cuối cùng giúp annotator không thể đoán được mẫu nào nằm ở cuối/đầu là mẫu QC.
    for annotator in annotators:
        ann_tasks = final_splits[annotator]
        random.shuffle(ann_tasks)
        
        # Chuyển đổi định dạng predictions -> annotations và cập nhật from_name từ 'ner' sang 'label'
        for t in ann_tasks:
            # Chuyển predictions -> annotations để annotator gán/sửa trực tiếp
            if "predictions" in t:
                preds = t.pop("predictions")
                ann_list = []
                for p in preds:
                    ann_list.append({
                        "result": p.get("result", []),
                        "was_cancelled": False,
                        "ground_truth": False
                    })
                t["annotations"] = ann_list
            
            # Đồng bộ from_name = 'label' khớp với XML config
            anns = t.get("annotations", [])
            for ann_item in anns:
                for res in ann_item.get("result", []):
                    if res.get("from_name") == "ner":
                        res["from_name"] = "label"
            
            # Sinh dữ liệu XML động khớp 100% với label_studio_config.xml
            labels_for_task = build_label_set(t["data"])
            t["data"]["labels_for_task"] = labels_for_task
            t["data"]["hotkey_hint"] = build_hotkey_hint(labels_for_task)
            t["data"]["context_header"] = build_context_header(t["data"])
        
        # Ghi file JSON
        out_json_path = output_dir / f"{annotator}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(ann_tasks, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Đã xuất file: {out_json_path.name} với {len(ann_tasks)} tasks "
            f"(dạng Annotation + from_name='label' + XML data động)"
        )

    # 7. In thống kê cuối chương trình
    print("\n" + "="*50)
    print("=== THỐNG KÊ CHI TIẾT BLIND DOUBLE ANNOTATION ===")
    print("="*50)
    print(f"Total tasks: {total_tasks}")
    print("\nPrimary:")
    for annotator in annotators:
        print(f"  {annotator}: {len(primary_splits[annotator])}")
        
    print("\nQC Rotation:")
    print(f"  A -> B: {len(qc_splits['A'])}")
    print(f"  B -> C: {len(qc_splits['B'])}")
    print(f"  C -> D: {len(qc_splits['C'])}")
    print(f"  D -> A: {len(qc_splits['D'])}")
    
    print("\nFinal:")
    for annotator in annotators:
        print(f"  {annotator}: {len(final_splits[annotator])}")
        
    total_duplicated = sum(len(qc_splits[a]) for a in annotators)
    total_final_tasks = sum(len(final_splits[a]) for a in annotators)
    real_overlap_rate = (total_duplicated / total_tasks) * 100
    
    print(f"\nUnique tasks: {total_tasks}")
    print(f"Duplicated tasks: {total_duplicated}")
    print(f"Overlap rate: {real_overlap_rate:.1f}%")
    print("="*50 + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Chia dữ liệu annotation theo phương pháp blind double annotation."
    )
    p.add_argument("--input", "-i", type=Path, default=Path("campaign_patched_all.json"),
                   help="Đường dẫn file JSON đầu vào (default: campaign_patched_all.json)")
    p.add_argument("--outdir", "-o", type=Path, default=Path("."),
                   help="Thư mục xuất kết quả (default: thư mục hiện tại)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed để tái lập kết quả (default: 42)")
    p.add_argument("--qc-ratio", type=float, default=0.15,
                   help="Tỉ lệ QC mẫu overlap (default: 0.15)")
    args = p.parse_args()

    if not args.input.exists():
        logger.error(f"File đầu vào không tồn tại: {args.input}")
        sys.exit(1)

    split_data(
        input_json=args.input,
        output_dir=args.outdir,
        seed=args.seed,
        qc_ratio=args.qc_ratio,
    )


if __name__ == "__main__":
    main()
