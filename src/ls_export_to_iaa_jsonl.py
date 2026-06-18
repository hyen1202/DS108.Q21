"""
ls_export_to_iaa_jsonl.py  —  Chuyển Label Studio JSON export → JSONL cho calculate_iaa.py
============================================================================================

Label Studio export format (mỗi task):
    {
        "id": <task_id>,
        "annotations": [
            {
                "result": [
                    {
                        "type": "labels",
                        "value": {
                            "start": <int>,
                            "end": <int>,
                            "text": <str>,
                            "labels": ["LABEL"]
                        }
                    }
                ]
            }
        ],
        "data": {
            "text": "<full text>",
            "meta": {
                "product_id": "...",
                ...
            }
        }
    }

Output JSONL format (mỗi dòng):
    {
        "id": "<product_id>",
        "text": "<full text>",
        "entities": [
            {"start_char": <int>, "end_char": <int>, "label": "<LABEL>", "text": "<span text>"}
        ]
    }

USAGE:
    python ls_export_to_iaa_jsonl.py --input YenTran_annotator_A.json --output YenTran_A.jsonl
    python ls_export_to_iaa_jsonl.py --input YenDo_Annotator_B.json   --output YenDo_B.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def convert_ls_export(input_path: Path, output_path: Path, annotator_name: str = "") -> dict:
    """
    Đọc Label Studio JSON export, chuyển sang JSONL format cho calculate_iaa.py.
    Dùng product_id (từ data.meta) làm sample ID để 2 annotators có thể match.

    Returns: dict thống kê
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"❌ File phải là JSON array (list). Got: {type(data).__name__}")
        sys.exit(1)

    stats = {
        "total_tasks": len(data),
        "tasks_with_annotation": 0,
        "tasks_no_annotation": 0,
        "tasks_empty_result": 0,
        "total_spans": 0,
        "duplicate_ids": 0,
        "written": 0,
    }

    seen_ids = {}  # product_id → task_id (để phát hiện duplicate)
    output_records = []

    for task in data:
        task_id = task.get("id", "?")
        task_data = task.get("data", {}) or {}
        text = task_data.get("text", "")
        meta = task_data.get("meta", {}) or {}
        product_id = str(meta.get("product_id") or meta.get("id") or task_id)  # fallback về id hoặc task_id nếu không có

        # Lấy annotation đầu tiên (chỉ 1 annotator per file)
        annotations = task.get("annotations", [])
        if not annotations:
            stats["tasks_no_annotation"] += 1
            # Vẫn ghi sample (text có thể không có entity nào)
            result_items = []
        else:
            stats["tasks_with_annotation"] += 1
            # Lấy annotation đầu tiên (không phải prediction)
            ann = annotations[0]
            result_items = ann.get("result", [])
            if not result_items:
                stats["tasks_empty_result"] += 1

        # Chuyển result items → entities
        entities = []
        for item in result_items:
            if item.get("type") != "labels":
                continue
            val = item.get("value", {})
            start = val.get("start")
            end = val.get("end")
            span_text = val.get("text", "")
            labels = val.get("labels", [])

            if start is None or end is None or not labels:
                continue

            # Label Studio dùng "labels" list, lấy cái đầu tiên
            label = labels[0]

            entities.append({
                "start_char": int(start),
                "end_char": int(end),
                "label": label,
                "text": span_text,
            })

        stats["total_spans"] += len(entities)

        # Kiểm tra duplicate product_id
        if product_id in seen_ids:
            stats["duplicate_ids"] += 1
            print(
                f"⚠️  Duplicate product_id='{product_id}' "
                f"(task {task_id} vs task {seen_ids[product_id]}). Ghi đè."
            )
        seen_ids[product_id] = task_id

        output_records.append({
            "id": product_id,
            "text": text,
            "entities": entities,
        })

    # Ghi ra file JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["written"] = len(output_records)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển Label Studio JSON export → JSONL cho calculate_iaa.py"
    )
    parser.add_argument("--input",  "-i", required=True, help="File JSON export từ Label Studio")
    parser.add_argument("--output", "-o", required=True, help="File JSONL output")
    parser.add_argument("--name",   "-n", default="",    help="Tên annotator (để in thống kê)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    name_label  = f"[{args.name}] " if args.name else ""

    if not input_path.exists():
        print(f"❌ File không tồn tại: {input_path}")
        sys.exit(1)

    print(f"📂 {name_label}Đọc: {input_path}")
    stats = convert_ls_export(input_path, output_path, args.name)

    print(f"\n✅ {name_label}Hoàn thành:")
    print(f"   Input:        {input_path}")
    print(f"   Output:       {output_path}")
    print(f"   Tổng tasks:   {stats['total_tasks']}")
    print(f"   Có annotation:{stats['tasks_with_annotation']}")
    print(f"   Không có ann: {stats['tasks_no_annotation']}")
    print(f"   Kết quả rỗng: {stats['tasks_empty_result']}")
    print(f"   Tổng spans:   {stats['total_spans']}")
    if stats["duplicate_ids"] > 0:
        print(f"   ⚠️  Duplicate product_ids: {stats['duplicate_ids']}")
    print(f"   Đã ghi:       {stats['written']} dòng → {output_path}")


if __name__ == "__main__":
    main()
