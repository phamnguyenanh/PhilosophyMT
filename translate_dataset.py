"""
IEP Dataset Translation Script — Parallel 2 Models
====================================================
Chạy song song 2 model, mỗi model xử lý file riêng (round-robin).

Yêu cầu:
    pip install openai python-dotenv

Cấu trúc .env:
    API_KEY=your_key_here
    ENDPOINT_URL=http://localhost:20128/v1
    MODEL_NAME_1=PhilosophyMT_1
    MODEL_NAME_2=PhilosophyMT_2

Chạy:
    python translate_dataset.py
    python translate_dataset.py --dir blank_dataset --delay 2.0
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

ENDPOINT_URL  = os.getenv("ENDPOINT_URL",   "http://localhost:20128/v1")
MODEL_NAME_1  = os.getenv("MODEL_NAME_1",   "PhilosophyMT_1")
MODEL_NAME_2  = os.getenv("MODEL_NAME_2",   "PhilosophyMT_2")
API_KEY       = os.getenv("API_KEY",        "no-key")

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = "blank_dataset"
REQUEST_DELAY       = 2.0   # Giây chờ giữa các file trong cùng một worker
MAX_RETRIES         = 3
RETRY_DELAY         = 5.0

# ─── Thread-safe logging & stats ──────────────────────────────────────────────
_print_lock = threading.Lock()
_stats_lock = threading.Lock()

_stats = {
    "skipped":    0,
    "translated": 0,
    "failed":     0,
    "records":    0,
}

def log(model_tag: str, msg: str) -> None:
    """In log có prefix model, thread-safe."""
    with _print_lock:
        print(f"{model_tag} {msg}", flush=True)

def update_stats(**kwargs) -> None:
    with _stats_lock:
        for k, v in kwargs.items():
            _stats[k] += v

# ─── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là một giáo sư triết học chuyên dịch thuật xuất sắc, \
có văn phong dịch thuật như các học giả lớn tại Việt Nam.

Nhiệm vụ của bạn: Dịch các đoạn văn triết học từ tiếng Anh sang tiếng Việt \
một cách chính xác, trung thành với nguyên bản và giữ nguyên tính học thuật.

Yêu cầu kiểm tra chất lượng (Crucial): Chú ý dịch đúng nghĩa triết học. \
Sau khi dịch xong toàn bộ các dòng, bạn phải đóng vai Biên tập viên rà soát \
ngược từ dòng cuối cùng lên dòng đầu tiên. Hãy sửa ngay nếu phát hiện đoạn \
nào ở cuối bị dịch lười, dịch tóm tắt hoặc dịch đại trà mất tính triết học \
trước khi hoàn thành."""

USER_PROMPT_TEMPLATE = """Dưới đây là {n} đoạn văn triết học cần dịch. \
Mỗi đoạn được đánh dấu bằng id số nguyên.

Sau khi rà soát chất lượng, hãy trả về KẾT QUẢ CUỐI CÙNG dưới dạng \
JSON object duy nhất với format sau (không có markdown, không giải thích thêm):
{{"<id>": "<bản dịch tiếng Việt>"}}

Các đoạn văn cần dịch:
{texts_json}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_records(filepath: Path) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(filepath: Path, records: list) -> None:
    # Ghi vào file tạm trước, rename sau để tránh mất dữ liệu nếu crash
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(filepath)


def needs_translation(records: list) -> list:
    return [r for r in records if not r.get("vi", "").strip()]


def repair_unescaped_quotes(text: str) -> str:
    """Sửa JSON bị vỡ do dấu nháy kép thẳng xuất hiện bên trong value."""
    parts = re.split(r'(?<=[{,\n])\s*"(\d+)"\s*:\s*"', text)
    if len(parts) < 3:
        return text
    pairs = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        key = parts[i]
        value_blob = parts[i + 1]
        end_pattern = re.search(r'"\s*,?\s*\n\s*(?="[\d]|\})', value_blob)
        if end_pattern:
            raw_value = value_blob[:end_pattern.start()]
        else:
            raw_value = value_blob.rstrip().rstrip('}').rstrip().rstrip('"').rstrip(',')
        raw_value = raw_value.replace('\\"', '\x00ESC\x00')
        raw_value = raw_value.replace('"', '\u201c\u201d')
        raw_value = raw_value.replace('\x00ESC\x00', '\\"')
        pairs.append((key, raw_value))
    items = ",\n".join(f'  "{k}": "{v}"' for k, v in pairs)
    return "{\n" + items + "\n}"


def extract_json_from_response(text: str) -> dict | None:
    text = text.strip()
    # 1. Parse thẳng
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Khối ```json```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. { ... } lớn nhất
    start, end = text.find("{"), text.rfind("}")
    candidate = ""
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 4. Sửa nháy kép không escaped
    try:
        repaired = repair_unescaped_quotes(candidate or text)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return None


def validate_translations(translations: dict, expected_ids: list) -> tuple:
    missing = [i for i in expected_ids
               if i not in translations or not str(translations[i]).strip()]
    return len(missing) == 0, missing


def call_translation_api(client: OpenAI, model_name: str,
                         records_to_translate: list, tag: str) -> dict | None:
    texts_input  = {str(r["id"]): r["en"] for r in records_to_translate}
    expected_ids = list(texts_input.keys())
    n = len(expected_ids)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        n=n,
        texts_json=json.dumps(texts_input, ensure_ascii=False, indent=2)
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(tag, f"  → Gọi API (lần {attempt}/{MAX_RETRIES}), {n} đoạn...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            raw_text = response.choices[0].message.content
            if not raw_text:
                log(tag, "  ✗ Response rỗng.")
            else:
                translations = extract_json_from_response(raw_text)
                if translations is None:
                    log(tag, f"  ✗ Không parse được JSON. Preview: {raw_text[:150]!r}")
                else:
                    ok, missing = validate_translations(translations, expected_ids)
                    if ok:
                        log(tag, f"  ✓ Nhận {len(translations)} bản dịch hợp lệ.")
                        return translations
                    else:
                        log(tag, f"  ✗ Thiếu {len(missing)} id: {missing[:5]}")
        except Exception as e:
            log(tag, f"  ✗ Lỗi API: {e}")

        if attempt < MAX_RETRIES:
            log(tag, f"  ⏳ Retry sau {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    return None


# ─── Worker — xử lý tuần tự một danh sách file ────────────────────────────────

def process_files(files: list, model_name: str, delay: float,
                  total_files: int) -> None:
    """
    Mỗi worker nhận danh sách file riêng (đã phân công round-robin từ main).
    Xử lý tuần tự, delay giữa các file.
    """
    tag = f"[{model_name}]"
    client = OpenAI(api_key=API_KEY, base_url=ENDPOINT_URL)

    for filepath in files:
        # Lấy số thứ tự file trong toàn bộ dataset (chỉ để hiển thị)
        num = re.search(r"\d+", filepath.name)
        idx = num.group() if num else "?"
        header = f"[{idx:>3}/{total_files}]"

        log(tag, f"{header} {filepath.name}")

        try:
            records = load_records(filepath)
        except Exception as e:
            log(tag, f"  ✗ Không đọc được: {e}")
            update_stats(failed=1)
            continue

        pending = needs_translation(records)
        if not pending:
            log(tag, f"  ✓ Đã dịch đầy đủ ({len(records)}) — bỏ qua.")
            update_stats(skipped=1)
            continue

        already = len(records) - len(pending)
        if already:
            log(tag, f"  ℹ Còn {len(pending)}/{len(records)} chưa dịch.")
        else:
            log(tag, f"  ℹ Chưa dịch ({len(pending)} records).")

        translations = call_translation_api(client, model_name, pending, tag)

        if translations is None:
            log(tag, f"  ✗ Thất bại sau {MAX_RETRIES} lần — bỏ qua.")
            update_stats(failed=1)
            time.sleep(delay)
            continue

        filled = 0
        for record in records:
            id_str = str(record["id"])
            if id_str in translations and translations[id_str].strip():
                record["vi"] = translations[id_str].strip()
                filled += 1

        try:
            save_records(filepath, records)
            log(tag, f"  ✓ Đã ghi {filled} bản dịch vào {filepath.name}")
            update_stats(translated=1, records=filled)
        except Exception as e:
            log(tag, f"  ✗ Lỗi ghi file: {e}")
            update_stats(failed=1)

        time.sleep(delay)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IEP Translator — 2 models parallel")
    parser.add_argument("--dir",   default=DEFAULT_DATASET_DIR)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                        help="Giây chờ giữa các file trong mỗi worker (default: 2.0)")
    args = parser.parse_args()

    dataset_dir = Path(args.dir)
    if not dataset_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {dataset_dir}")
        sys.exit(1)

    files = sorted(
        dataset_dir.glob("*.jsonl"),
        key=lambda p: int(re.search(r"\d+", p.name).group())
                      if re.search(r"\d+", p.name) else 0
    )
    if not files:
        print(f"[LỖI] Không có file .jsonl trong {dataset_dir}")
        sys.exit(1)

    total = len(files)

    # Phân file round-robin: file chẵn → model 1, file lẻ → model 2
    files_m1 = files[0::2]   # index 0, 2, 4, ...
    files_m2 = files[1::2]   # index 1, 3, 5, ...

    print(f"[INFO] Tìm thấy {total} file trong '{dataset_dir}'")
    print(f"[INFO] Endpoint  : {ENDPOINT_URL}")
    print(f"[INFO] Model 1   : {MODEL_NAME_1} — {len(files_m1)} file")
    print(f"[INFO] Model 2   : {MODEL_NAME_2} — {len(files_m2)} file")
    print(f"[INFO] Bắt đầu song song...\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_files, files_m1, MODEL_NAME_1, args.delay, total),
            executor.submit(process_files, files_m2, MODEL_NAME_2, args.delay, total),
        ]
        # Chờ cả 2 worker hoàn thành, re-raise nếu có exception
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print()
    print("=" * 55)
    print("[HOÀN THÀNH]")
    print(f"  Thời gian          : {mins}m {secs}s")
    print(f"  Tổng file          : {total}")
    print(f"  Đã dịch xong       : {_stats['translated']}")
    print(f"  Bỏ qua (đủ rồi)    : {_stats['skipped']}")
    print(f"  Thất bại            : {_stats['failed']}")
    print(f"  Tổng bản dịch mới  : {_stats['records']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
