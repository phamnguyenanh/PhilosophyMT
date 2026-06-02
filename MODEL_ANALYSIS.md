# Phân Tích Model Và Metric

Tài liệu này giải thích các mô hình, cách fine-tune, metric đánh giá và kết quả thực nghiệm của PhilosophyMT. README chỉ giữ phần hướng dẫn pipeline; phần diễn giải chi tiết nằm ở đây.

## 1. Bài Toán

Task của project là dịch văn bản triết học từ tiếng Anh sang tiếng Việt. Đây là domain khó hơn dịch phổ thông vì:

- Câu thường dài, nhiều mệnh đề phụ.
- Thuật ngữ triết học cần nhất quán.
- Cùng một câu có thể có nhiều bản dịch đúng về mặt ngữ nghĩa.
- Dịch sát chữ quá có thể khó đọc, nhưng dịch tự do quá có thể mất ý triết học.

Do đó, automatic metrics như BLEU và chrF++ chỉ nên xem là tín hiệu định lượng ban đầu. Đánh giá cuối cùng vẫn cần đọc mẫu dịch hoặc human evaluation.

## 2. MarianMT

MarianMT là nhóm mô hình machine translation được tích hợp trong Hugging Face Transformers. Về kiến trúc, MarianMT là mô hình encoder-decoder sequence-to-sequence:

- Encoder đọc câu nguồn.
- Decoder sinh câu đích từng token.
- Mô hình được huấn luyện trực tiếp cho bài toán dịch máy.

Điểm quan trọng là MarianMT sinh bản dịch theo cơ chế seq2seq chuyên dụng, không cần prompt dạng chat. Với dữ liệu song ngữ có cặp `source`/`target` rõ ràng, kiến trúc này thường rất phù hợp.

Trong project này, model được dùng là:

```text
Helsinki-NLP/opus-mt-en-vi
```

Đây là model OPUS-MT cho hướng dịch English → Vietnamese. Việc fine-tune model này trên dataset triết học giúp model thích nghi với domain chuyên ngành: thuật ngữ, văn phong học thuật và cấu trúc câu đặc thù.

Tham khảo:

- Hugging Face MarianMT documentation: https://huggingface.co/docs/transformers/model_doc/marian
- Hugging Face model page: https://huggingface.co/Helsinki-NLP/opus-mt-en-vi

## 3. Helsinki-NLP/opus-mt-en-vi

`Helsinki-NLP/opus-mt-en-vi` là pretrained translation model cho cặp ngôn ngữ Anh-Việt. Vì đã được pretrain cho đúng hướng dịch, model có lợi thế lớn so với language model tổng quát:

- Không cần học từ đầu cách chuyển từ tiếng Anh sang tiếng Việt.
- Tokenizer và decoder đã được tối ưu cho translation.
- Fine-tune chỉ cần điều chỉnh model theo domain triết học.

Trong notebook Marian, các input được tokenize bằng:

```python
tokenizer(en)["input_ids"]
```

Target tiếng Việt được tokenize đúng chế độ target:

```python
tokenizer(text_target=vi)["input_ids"]
```

Đây là điểm quan trọng với seq2seq tokenizer: source và target có thể dùng logic xử lý khác nhau.

## 4. Qwen/Qwen2.5-0.5B-Instruct

`Qwen/Qwen2.5-0.5B-Instruct` là instruction-tuned causal language model. Khác với MarianMT, Qwen không phải model dịch máy encoder-decoder chuyên dụng. Nó là causal LM:

- Nhận một chuỗi prompt đầu vào.
- Sinh token tiếp theo theo cơ chế autoregressive.
- Dùng chat template để mô phỏng hội thoại `system` / `user` / `assistant`.

Trong project này, mỗi example được format thành instruction:

```text
System: You are a professional English-to-Vietnamese translator...
User: <English source>
Assistant: <Vietnamese translation>
```

Khi inference, model nhận system prompt + user English text, sau đó sinh phần assistant output là bản dịch tiếng Việt.

Vì Qwen2.5-0.5B-Instruct chỉ có khoảng 0.5B parameters, nó nhẹ và phù hợp để thử nghiệm trên Colab/T4, nhưng năng lực dịch chuyên ngành ban đầu thấp hơn model dịch máy chuyên dụng. Fine-tune bằng QLoRA giúp cải thiện mạnh.

Tham khảo:

- Hugging Face model page: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

## 5. LoRA, QLoRA Và Adapter

LoRA là kỹ thuật fine-tune hiệu quả tham số. Thay vì cập nhật toàn bộ trọng số của model, LoRA thêm các ma trận nhỏ vào một số layer chính, rồi chỉ train các ma trận đó.

QLoRA kết hợp LoRA với quantization:

- Base model được load ở 4-bit để tiết kiệm VRAM.
- LoRA adapter vẫn được train để học task mới.
- Sau training, artifact chính là adapter, không phải full model.

Trong project này, Qwen được fine-tune bằng QLoRA vì:

- Giảm VRAM so với full fine-tune.
- Phù hợp với Colab GPU phổ thông.
- Dễ lưu adapter riêng tại `runs/qwen25_0_5b_qlora/adapter/`.

Adapter có thể hiểu là phần trọng số bổ sung đã học sau fine-tune. Khi muốn dùng model fine-tuned, cần load base model Qwen rồi gắn adapter.

## 6. BLEU

BLEU đo mức độ trùng khớp n-gram giữa bản dịch của model và bản dịch reference. Ý tưởng chính:

- So sánh các cụm 1-gram, 2-gram, 3-gram, 4-gram.
- Tính precision của các n-gram được model sinh ra.
- Có brevity penalty để phạt bản dịch quá ngắn.

Ưu điểm:

- Phổ biến trong machine translation.
- Dễ so sánh giữa các run.
- Nhạy với độ giống bề mặt so với reference.

Hạn chế:

- Không hiểu ngữ nghĩa sâu.
- Phạt các cách diễn đạt đúng nhưng khác reference.
- Với văn bản triết học, BLEU có thể đánh giá thấp bản dịch đúng nhưng diễn đạt khác.

## 7. chrF++

chrF++ đo overlap ở mức character n-gram và có thêm word n-gram. So với BLEU, chrF++ thường mềm hơn với các biến thể hình thái và cách viết.

Ưu điểm:

- Hữu ích cho ngôn ngữ có biến thể từ vựng/hình thái.
- Ít cứng hơn BLEU vì dùng character-level matching.
- Trong dịch Anh-Việt, chrF++ thường phản ánh độ gần bề mặt tốt hơn khi từ/cụm từ có biến thể nhỏ.

Hạn chế:

- Vẫn không thực sự hiểu nghĩa.
- Có thể cho điểm tương đối cao nếu bản dịch giống ký tự nhưng sai sắc thái triết học.
- Không thay thế được đánh giá thủ công.

Trong project này, BLEU và chrF++ được dùng cùng nhau. BLEU cho tín hiệu n-gram ở mức từ/cụm từ, chrF++ bổ sung tín hiệu mềm hơn ở mức ký tự.

## 8. Kết Quả Thực Nghiệm

Kết quả lấy từ `runs/*/eval_metrics.json`. Tất cả model được đánh giá trên cùng test split 304 samples.

| Rank | Run | Model | Method | LR | Test BLEU | Test chrF++ |
|---:|---|---|---|---:|---:|---:|
| 1 | `marian_lr5e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `5e-5` | 48.4134 | 65.8429 |
| 2 | `marian_lr3e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `3e-5` | 45.8165 | 64.0285 |
| 3 | `qwen25_0_5b_qlora` | `Qwen/Qwen2.5-0.5B-Instruct` | QLoRA 4-bit | `2e-4` | 44.1168 | 62.1533 |
| 4 | `marian_lr1e-5` | `Helsinki-NLP/opus-mt-en-vi` | full fine-tune | `1e-5` | 39.3259 | 59.0951 |
| 5 | `marian_baseline` | `Helsinki-NLP/opus-mt-en-vi` | baseline | - | 11.6533 | 31.1529 |
| 6 | `qwen25_0_5b_base` | `Qwen/Qwen2.5-0.5B-Instruct` | base no fine-tune | - | 8.6148 | 27.3461 |

Validation results:

| Run | Val BLEU | Val chrF++ |
|---|---:|---:|
| `marian_lr5e-5` | 47.5904 | 64.3737 |
| `marian_lr3e-5` | 45.3846 | 62.6376 |
| `qwen25_0_5b_qlora` | 43.4027 | 60.7321 |
| `marian_lr1e-5` | 39.8820 | 58.4052 |
| `qwen25_0_5b_base` | 10.7835 | 29.5578 |

`marian_baseline` hiện chỉ có test metrics trong artifact.

## 9. Phân Tích Kết Quả

### Marian fine-tuning rất hiệu quả

Marian baseline đạt Test BLEU `11.6533` và chrF++ `31.1529`. Sau fine-tune, Marian LR `5e-5` đạt Test BLEU `48.4134` và chrF++ `65.8429`.

Mức tăng này cho thấy domain adaptation trên dataset triết học có tác động rất lớn. Pretrained translation model đã biết dịch Anh-Việt, còn fine-tune giúp model học văn phong và thuật ngữ của domain.

### Learning rate `5e-5` tốt nhất trong sweep hiện tại

Ba learning rate Marian cho kết quả:

```text
1e-5 -> Test BLEU 39.3259, chrF++ 59.0951
3e-5 -> Test BLEU 45.8165, chrF++ 64.0285
5e-5 -> Test BLEU 48.4134, chrF++ 65.8429
```

Trong sweep này, `5e-5` tốt nhất theo cả BLEU và chrF++. Tuy nhiên, chỉ mới thử ba giá trị, nên chưa thể kết luận đây là learning rate tối ưu toàn cục.

### Qwen base yếu, nhưng QLoRA cải thiện mạnh

Qwen2.5-0.5B base chưa fine-tune đạt Test BLEU `8.6148`, chrF++ `27.3461`, thấp hơn cả Marian baseline. Sau QLoRA, kết quả tăng lên Test BLEU `44.1168`, chrF++ `62.1533`.

Điều này cho thấy Qwen có khả năng học format và domain dịch khi được fine-tune, nhưng bản base instruct 0.5B không tự nhiên mạnh cho dịch chuyên ngành Anh-Việt nếu không có adapter.

### Marian vẫn phù hợp hơn cho task này

Qwen QLoRA gần Marian LR `3e-5`, nhưng vẫn thấp hơn Marian LR `5e-5`:

```text
Qwen QLoRA      -> Test BLEU 44.1168, chrF++ 62.1533
Marian LR 3e-5  -> Test BLEU 45.8165, chrF++ 64.0285
Marian LR 5e-5  -> Test BLEU 48.4134, chrF++ 65.8429
```

Với dataset song ngữ có cặp source-target rõ ràng, MarianMT encoder-decoder có lợi thế kiến trúc cho machine translation. Qwen causal LM cần prompt/chat format và generation autoregressive, nên chi phí inference cao hơn và metric thấp hơn trong cấu hình 0.5B QLoRA hiện tại.

