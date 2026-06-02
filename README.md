# PhilosophyMT

PhilosophyMT là dự án xây dựng, fine-tune và đánh giá các mô hình dịch Anh-Việt cho văn bản triết học. Dữ liệu được lấy từ Internet Encyclopedia of Philosophy (IEP), sau đó được tách đoạn, dịch/gán nhãn tiếng Việt, và dùng để fine-tune các mô hình machine translation.

Mục tiêu chính:

- Tạo dataset song ngữ Anh-Việt cho văn bản triết học.
- Fine-tune mô hình dịch chuyên ngành trên domain triết học.
- So sánh MarianMT encoder-decoder với Qwen causal LM QLoRA.
- Đánh giá bằng BLEU và chrF++ trên cùng một split theo topic.

Phần phân tích chi tiết về mô hình, metric và kết quả thực nghiệm nằm trong [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md).

## Cấu Trúc Repository

```text
.
|-- dataset.jsonl                         # Dataset song ngữ đã hoàn thiện
|-- blank_dataset.jsonl                   # Dataset scrape từ IEP, chưa có bản dịch
|-- link_IEP.txt                          # Danh sách URL IEP đầu vào
|-- scrape_iep.py                         # Scrape và chunk văn bản IEP
|-- split_blank_to_labeling.py            # Tách blank_dataset.jsonl thành nhiều file nhỏ
|-- translate_dataset.py                  # Dịch các file con qua OpenAI-compatible API
|-- merge_dataset.py                      # Gộp các file đã dịch thành dataset.jsonl
|-- notebooks/
|   |-- 01_eda_dataset.ipynb
|   |-- 02_finetune_marian_lr_sweep.ipynb
|   |-- 03_finetune_qwen25_0_5b_qlora.ipynb
|   |-- 04_compare_all_models.ipynb
|-- runs/                                 # Artifact local, bị ignore bởi git
|   |-- splits/topic_split_seed42.json
|   |-- marian_baseline/
|   |-- marian_lr1e-5/
|   |-- marian_lr3e-5/
|   |-- marian_lr5e-5/
|   |-- qwen25_0_5b_base/
|   |-- qwen25_0_5b_qlora/
```

`runs/`, checkpoints, model artifacts và `.env` không nên commit lên Git. `runs/` chỉ là thư mục artifact local để lưu metrics, predictions, checkpoints và model sau khi train.

## Dataset

Dataset chính là `dataset.jsonl`. Mỗi dòng là một JSON record:

```json
{
  "id": 1,
  "topic": "...",
  "en": "English source paragraph...",
  "vi": "Vietnamese translation..."
}
```

Split train/validation/test được chia theo `topic`, không chia random từng dòng. Cách chia này giúp giảm leakage: các đoạn trong cùng một chủ đề không bị rơi đồng thời vào train và test.

Split hiện tại được lưu tại:

```text
runs/splits/topic_split_seed42.json
```

Tất cả các model trong bảng kết quả bên dưới được đánh giá trên cùng test split gồm 304 samples.

## Cấu Hình Môi Trường

Tạo file `.env` từ file mẫu:

```bash
cp .env.example .env
```

Các biến môi trường được dùng bởi `translate_dataset.py`:

```env
API_KEY=your_api_key_here
ENDPOINT_URL=http://localhost:20128/v1
MODEL_NAME_1=PhilosophyMT_1
MODEL_NAME_2=PhilosophyMT_2
```

`ENDPOINT_URL` là endpoint OpenAI-compatible. Có thể là server local, tunnel, hoặc endpoint dịch nội bộ.

## Pipeline

1. Scrape IEP:

```bash
python scrape_iep.py
```

Script đọc URL từ `link_IEP.txt`, tải HTML, lấy nội dung bài viết, bỏ các section như table of contents, references, author information, sau đó chunk thành các đoạn ngắn hơn để đưa vào dataset.

2. Tách dataset chưa dịch thành các phần nhỏ:

```bash
python split_blank_to_labeling.py
```

Mặc định script tách `blank_dataset.jsonl` vào thư mục `blank_dataset/`, mỗi file con khoảng 30 dòng.

3. Dịch/gán nhãn các file con:

```bash
python translate_dataset.py
```

Script sử dụng OpenAI-compatible API endpoint từ `.env`, hỗ trợ 2 model/worker chạy song song. Các record chưa có `vi` sẽ được dịch và ghi lại vào file.

4. Gộp dataset:

```bash
python merge_dataset.py
```

Script gộp các file `.jsonl` trong `blank_dataset/` thành `dataset.jsonl`.

5. EDA và chọn max length:

```text
notebooks/01_eda_dataset.ipynb
```

Notebook phân tích word count, token length theo Marian tokenizer, truncation risk và đề xuất `MAX_SOURCE_LEN` / `MAX_TARGET_LEN`.

6. Fine-tune MarianMT:

```text
notebooks/02_finetune_marian_lr_sweep.ipynb
```

Notebook train Marian baseline và LR sweep cho `Helsinki-NLP/opus-mt-en-vi`.

7. Fine-tune Qwen QLoRA:

```text
notebooks/03_finetune_qwen25_0_5b_qlora.ipynb
```

Notebook fine-tune `Qwen/Qwen2.5-0.5B-Instruct` bằng QLoRA 4-bit.

8. So sánh model:

```text
notebooks/04_compare_all_models.ipynb
```

Notebook đọc `runs/*/eval_metrics.json` và `runs/*/predictions_test.jsonl`, sau đó tạo bảng so sánh BLEU/chrF++ và hiển thị mẫu dịch.

## Notebooks

| Notebook | Vai trò |
|---|---|
| `01_eda_dataset.ipynb` | EDA dataset, word length, Marian tokenizer length, truncation risk |
| `02_finetune_marian_lr_sweep.ipynb` | Marian baseline và fine-tune với LR `1e-5`, `3e-5`, `5e-5` |
| `03_finetune_qwen25_0_5b_qlora.ipynb` | Qwen2.5-0.5B-Instruct baseline và QLoRA fine-tune |
| `04_compare_all_models.ipynb` | Tổng hợp metrics, vẽ biểu đồ, xem mẫu dịch |

## Kết Quả Tóm Tắt

Kết quả lấy từ `runs/*/eval_metrics.json`. Metric chính là BLEU và chrF++.

| Rank | Run | Model | Method | LR | Test BLEU | Test chrF++ |
|---:|---|---|---|---:|---:|---:|
| 1 | `marian_lr5e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `5e-5` | 48.4134 | 65.8429 |
| 2 | `marian_lr3e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `3e-5` | 45.8165 | 64.0285 |
| 3 | `qwen25_0_5b_qlora` | `Qwen/Qwen2.5-0.5B-Instruct` | QLoRA 4-bit | `2e-4` | 44.1168 | 62.1533 |
| 4 | `marian_lr1e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `1e-5` | 39.3259 | 59.0951 |
| 5 | `marian_baseline` | `Helsinki-NLP/opus-mt-en-vi` | baseline | - | 11.6533 | 31.1529 |
| 6 | `qwen25_0_5b_base` | `Qwen/Qwen2.5-0.5B-Instruct` | base no fine-tune | - | 8.6148 | 27.3461 |

Đọc phân tích chi tiết trong [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md).

## Cách Reproduce

### 1. Cài dependencies cơ bản

```bash
pip install transformers datasets evaluate sacrebleu accelerate sentencepiece pandas matplotlib
```

Cho Qwen QLoRA:

```bash
pip install bitsandbytes peft trl
```

Cho scripts scrape/dịch:

```bash
pip install requests beautifulsoup4 openai python-dotenv
```

### 2. Chuẩn bị dataset

Nếu muốn tạo lại dataset từ đầu:

```bash
python scrape_iep.py
python split_blank_to_labeling.py
python translate_dataset.py
python merge_dataset.py
```

Nếu `dataset.jsonl` đã có sẵn, có thể bỏ qua bước scrape/dịch và chạy thẳng notebooks.

### 3. Chạy notebooks theo thứ tự

```text
notebooks/01_eda_dataset.ipynb
notebooks/02_finetune_marian_lr_sweep.ipynb
notebooks/03_finetune_qwen25_0_5b_qlora.ipynb
notebooks/04_compare_all_models.ipynb
```

Các notebooks đều có helper để resolve root path:

```text
/content/drive/MyDrive/PhilosophyMT
```

khi chạy trên Colab, hoặc repo root nếu chạy local.

## Notes

- `runs/` bị ignore trong git, nên kết quả training/evaluation là artifact local.
- `.env` không được commit. Dùng `.env.example` để biết cấu trúc biến môi trường.
- COMET đã bị bỏ khỏi notebook compare do dependency conflict. Nếu muốn dùng COMET, nên tạo environment riêng tách khỏi training notebooks.
- BLEU/chrF++ là automatic metrics, không thay thế hoàn toàn human evaluation cho dịch triết học.
