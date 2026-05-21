"""
IEP (Internet Encyclopedia of Philosophy) Scraper
==================================================
Đọc Links.txt → cào bài viết → chunking → xuất dataset_blank.jsonl

Yêu cầu:
    pip install requests beautifulsoup4

Chạy:
    python scrape_iep.py
    python scrape_iep.py --links Links.txt --output dataset_blank.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ─── Cấu hình mặc định ────────────────────────────────────────────────────────
DEFAULT_LINKS_FILE  = "link_IEP.txt"
DEFAULT_OUTPUT_FILE = "blank_dataset.jsonl"

MIN_WORDS          = 50    # Đoạn văn ít hơn ngưỡng này → bỏ qua
MAX_WORDS          = 150   # Đoạn văn nhiều hơn ngưỡng này → tách đôi
OVERLAP_SENTENCES  = 1     # Số câu overlap khi tách đoạn văn dài
REQUEST_DELAY      = 1.5   # Giây chờ giữa các request (tránh bị block)
REQUEST_TIMEOUT    = 25    # Timeout cho mỗi request
MAX_RETRIES        = 3     # Số lần thử lại khi gặp lỗi mạng

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# ─── Bước 1: Đọc và lọc link ─────────────────────────────────────────────────

def load_unique_links(filepath: str) -> list:
    """Đọc file, loại bỏ dòng trống và URL trùng lặp, giữ thứ tự xuất hiện đầu tiên."""
    seen  = set()
    links = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url and url.startswith("http") and url not in seen:
                seen.add(url)
                links.append(url)
    print(f"[INFO] Đã tải {len(links)} link duy nhất từ '{filepath}'.")
    return links


# ─── Bước 2: Cào HTML ─────────────────────────────────────────────────────────

def fetch_html(url: str):
    """Tải HTML với retry. Trả về None nếu thất bại."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            print(f"  [HTTP {code}] lần thử {attempt}/{MAX_RETRIES}")
            if code in (403, 404, 410):
                return None   # Lỗi vĩnh viễn → không retry
        except Exception as e:
            print(f"  [LỖI] {e} (lần thử {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * attempt)
    return None


def extract_title_and_body(html: str):
    """
    Phân tích HTML theo cấu trúc thực tế của IEP:

    Cấu trúc trang IEP:
      div.entry-content
        h1                          ← topic / tiêu đề bài
        p p p                       ← đoạn mở đầu  (CẦN LẤY)
        h3 "Table of Contents"      ← ToC           (BỎ QUA + nội dung liền sau)
        p p p                       ← nội dung      (CẦN LẤY)
        h2 (tên section)            ← heading       (bỏ qua heading, lấy p bên trong)
        h3 (tên sub-section)        ← heading       (bỏ qua heading, lấy p bên trong)
        p p p                       ← nội dung      (CẦN LẤY)
        ...
        h2 "References and Further Reading"  ← DỪNG tại đây
        h2/h3 "Author Information"           ← BỎ QUA

    Chiến lược: duyệt tuần tự các direct-children của entry-content bằng
    state machine; chỉ thu thập <p> khi trạng thái là COLLECTING.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Tiêu đề: h1 đầu tiên trong entry-content ───────────────────────
    title = ""
    content_div = (
        soup.find("div", class_=re.compile(r"entry-content|post-content|article-content|article-body", re.I))
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", id=re.compile(r"^content$|^main$", re.I))
        or soup.body
    )
    if not content_div:
        return "", ""

    h1 = content_div.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title:
        t = soup.find("title")
        if t:
            title = t.get_text(" ", strip=True).split("|")[0].strip()

    # ── 2. Keywords nhận biết các heading cần dừng / bỏ qua ──────────────
    STOP_H2_KEYWORDS = (          # gặp h2 này → dừng hoàn toàn
        "references and further reading",
        "references",
        "bibliography",
        "further reading",
    )
    SKIP_HEADING_KEYWORDS = (     # gặp h2/h3 này → bỏ qua section đó
        "table of contents",
        "contents",
        "author information",
        "about the author",
        "see also",
        "related entries",
        "acknowledgements",
    )

    # ── 3. Duyệt tuần tự bằng state machine ──────────────────────────────
    # Trạng thái:
    #   COLLECTING  – đang thu thập <p>
    #   SKIPPING    – đang bỏ qua section (ToC, Author Info…) cho tới h2/h3 khác
    #   DONE        – đã gặp References → dừng hẳn

    STATE_COLLECTING = "collecting"
    STATE_SKIPPING   = "skipping"
    STATE_DONE       = "done"

    state      = STATE_COLLECTING
    paragraphs = []

    def heading_text(tag):
        return tag.get_text(" ", strip=True).lower()

    def is_stop_h2(tag):
        txt = heading_text(tag)
        return any(kw in txt for kw in STOP_H2_KEYWORDS)

    def is_skip_heading(tag):
        txt = heading_text(tag)
        return any(kw in txt for kw in SKIP_HEADING_KEYWORDS)

    # Duyệt mọi thẻ con (đệ quy) theo thứ tự xuất hiện trong DOM.
    # Dùng find_all để có đủ h2, h3, p ở mọi cấp nesting.
    for tag in content_div.find_all(["h1", "h2", "h3", "h4", "p"]):

        if state == STATE_DONE:
            break

        name = tag.name

        # ── h1: bỏ qua (đã lấy title ở trên) ───────────────────────────
        if name == "h1":
            continue

        # ── h2: kiểm tra dừng hoặc skip ─────────────────────────────────
        if name == "h2":
            if is_stop_h2(tag):
                state = STATE_DONE
                break
            elif is_skip_heading(tag):
                state = STATE_SKIPPING
            else:
                # h2 bình thường (tên section) → tiếp tục collecting
                state = STATE_COLLECTING
            continue

        # ── h3 / h4: kiểm tra skip ───────────────────────────────────────
        if name in ("h3", "h4"):
            if is_skip_heading(tag):
                state = STATE_SKIPPING
            else:
                # h3/h4 bình thường (sub-section heading) → tiếp tục collecting
                # KHÔNG đổi state về collecting nếu đang skipping do h2 skip
                if state != STATE_SKIPPING:
                    state = STATE_COLLECTING
                # Nếu đang skipping nhưng gặp h3 không phải skip-keyword
                # → đây là sub-section của section đang skip, vẫn giữ skipping.
                # Tuy nhiên nếu muốn collect lại thì bỏ comment dưới:
                # state = STATE_COLLECTING
            continue

        # ── p: thu thập nếu đang collecting ─────────────────────────────
        if name == "p" and state == STATE_COLLECTING:
            # Bỏ qua p rỗng hoặc chỉ có link điều hướng (thường < 8 từ)
            text = tag.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

    raw_text = "\n\n".join(paragraphs)
    return title, raw_text


# ─── Bước 3: Tách đoạn văn với bộ lọc ────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def split_into_sentences(text: str) -> list:
    """
    Tách văn bản thành danh sách câu.
    Dùng re.split tại ranh giới kết thúc câu (.!?) + khoảng trắng + chữ hoa/ngoặc.
    Các phần quá ngắn được gộp vào câu trước (tránh nhận viết tắt là ranh giới câu).
    """
    raw_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\(\'])', text)
    merged = []
    buf = ""
    for part in raw_parts:
        buf = (buf + " " + part).strip() if buf else part
        if word_count(buf) >= 5:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = (merged[-1] + " " + buf).strip()
        else:
            merged.append(buf)
    return merged


def chunk_paragraph(para: str, overlap: int = OVERLAP_SENTENCES) -> list:
    """
    Tách đoạn văn dài thành các chunk <= MAX_WORDS bằng vòng lặp iterative.

    Thuật toán:
      - Dùng queue; mỗi lần lấy ra một đoạn cần xử lý.
      - Nếu đoạn <= MAX_WORDS → giữ nguyên (done).
      - Nếu không tách được câu (n <= 2) hoặc tách không làm giảm kích thước
        (vô hạn) → giữ nguyên, tránh đệ quy vô tận.
      - Ngược lại: tách đôi tại điểm giữa câu + overlap, đưa 2 nửa vào queue.
    """
    result  = []
    # Queue chứa (đoạn_văn, số_lần_đã_thử_tách)
    queue   = [(para, 0)]
    MAX_SPLITS = 20   # Tối đa 20 lần tách bất kỳ một nhánh → tránh vòng lặp vô hạn

    while queue:
        current, depth = queue.pop(0)

        # Đã đủ nhỏ hoặc đã thử quá nhiều lần → dừng
        if word_count(current) <= MAX_WORDS or depth >= MAX_SPLITS:
            result.append(current)
            continue

        sentences = split_into_sentences(current)
        n = len(sentences)

        # Không đủ câu để tách hoặc chỉ có 1 câu cực dài → giữ nguyên
        if n <= 2:
            result.append(current)
            continue

        mid = n // 2
        right_overlap = sentences[mid : mid + overlap]
        left_overlap  = sentences[max(0, mid - overlap) : mid]

        first_half  = " ".join(sentences[:mid] + right_overlap)
        second_half = " ".join(left_overlap + sentences[mid:])

        # Kiểm tra tiến triển: nếu tách không làm giảm kích thước thì dừng
        if word_count(first_half) >= word_count(current) or \
           word_count(second_half) >= word_count(current):
            result.append(current)
            continue

        queue.append((first_half,  depth + 1))
        queue.append((second_half, depth + 1))

    return result


def process_raw_text(raw_text: str) -> list:
    """
    Từ văn bản thô:
      1. Tách thành block theo dòng trống
      2. Lọc theo số từ (MIN_WORDS / MAX_WORDS)
      3. Chunk các block dài
    """
    blocks = re.split(r"\n{2,}", raw_text)
    final_chunks = []

    for block in blocks:
        block = re.sub(r"[ \t]+", " ", block).strip()
        block = re.sub(r"\n+",    " ", block)

        if not block:
            continue

        wc = word_count(block)

        if wc < MIN_WORDS:
            continue
        elif wc <= MAX_WORDS:
            final_chunks.append(block)
        else:
            final_chunks.extend(chunk_paragraph(block))

    return [c for c in final_chunks if word_count(c) >= MIN_WORDS]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IEP Article Scraper → JSONL Dataset")
    parser.add_argument("--links",  default=DEFAULT_LINKS_FILE,  help="File chứa danh sách URL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="File đầu ra .jsonl")
    args = parser.parse_args()

    if not Path(args.links).exists():
        print(f"[LỖI] Không tìm thấy file: {args.links}")
        sys.exit(1)

    links = load_unique_links(args.links)
    total_links = len(links)

    record_id = 1
    skipped   = 0
    out_path  = Path(args.output)
    out_path.unlink(missing_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, url in enumerate(links, 1):
            print(f"\n[{i:>3}/{total_links}] {url}")

            html = fetch_html(url)
            if html is None:
                print("  → Bỏ qua (không tải được HTML).")
                skipped += 1
                time.sleep(REQUEST_DELAY)
                continue

            title, raw_text = extract_title_and_body(html)

            if not raw_text.strip():
                print("  → Bỏ qua (không trích xuất được nội dung).")
                skipped += 1
                time.sleep(REQUEST_DELAY)
                continue

            chunks = process_raw_text(raw_text)

            if not chunks:
                print(f"  → Bỏ qua (không có đoạn văn hợp lệ). Title: {title!r}")
                skipped += 1
                time.sleep(REQUEST_DELAY)
                continue

            print(f"  ✓ Title: {title!r}  |  Số đoạn văn: {len(chunks)}")

            for chunk in chunks:
                record = {
                    "id":    record_id,
                    "topic": title,
                    "en":    chunk,
                    "vi":    "",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_id += 1

            time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print(f"[HOÀN THÀNH]")
    print(f"  Link xử lý thành công : {total_links - skipped}/{total_links}")
    print(f"  Bản ghi đã xuất       : {record_id - 1}")
    print(f"  File đầu ra            : {out_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
