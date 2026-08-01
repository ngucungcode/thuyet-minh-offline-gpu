# Phase 3 — Báo cáo dịch offline

Ngày nghiệm thu: 01/08/2026<br>
Máy nghiệm thu: NVIDIA GeForce RTX 3090 24 GiB, driver 580.82.07

## Kết luận

**PASS trên GPU VM native.** Pipeline đã đi từ `READY_TRANSLATION` qua
`TRANSLATING` đến `READY_TTS`, có checkpoint từng block và artifact JSON atomic.
Đầu vào tiếng Việt bypass model. Model dịch mặc định là Gemma 4 31B Q4; Gemma 4
E2B Q4 là lựa chọn cục bộ nhanh hơn. Worker không tải model khi suy luận.

Phase 4 (TTS, fit thời lượng và mux MP4) chưa được triển khai trong cổng này.

## Thành phần đã khóa

| Thành phần | Pin | Artifact |
|---|---|---|
| llama.cpp CUDA | `b10208`, commit `9d9a6d29f6b981cc7f41983d26e56485c6af1811` | CUDA sm_86, `LLAMA_CURL=OFF`, Web UI tắt |
| Gemma 4 E2B IT QAT Q4_0 | revision `675cff42a74c774d6cb76f76d8eacb49b48c9b93` | 3.349.516.256 byte, tree SHA `fc8aa4b40a3aa26ea031cf5a553ed993cbef952afeefad7f935ddd2aa69bc182` |
| Gemma 4 31B IT QAT Q4_0 | revision `59dde24573e7e61570dba08b18a2e1fe246955ed` | 17.651.001.568 byte, tree SHA `3c225a6602607e730121efba31931812a58ca3fad4787229e73e93f8041ce610` |

Hai model được tải bằng model-manager có egress riêng, kiểm size/SHA-256 đầy đủ,
publish atomic và có receipt. `llama-server` chỉ nhận đường dẫn GGUF đã xác minh.

## Kết quả đo thực tế

| Kiểm tra | Kết quả |
|---|---|
| Build CUDA và nhận GPU | PASS — RTX 3090, 24.124 MiB khả dụng theo llama.cpp |
| Gemma 4 E2B load | 1,61 giây; khoảng 1.780 MiB VRAM |
| Gemma 4 31B load | 2,83 giây; khoảng 18.674 MiB VRAM |
| RSS tiến trình 31B | khoảng 2,89 GB |
| Full hash model 31B trong adapter production | 11,27 giây |
| Adapter production 31B startup | 3,09 giây |
| Full acceptance 31B, 6 câu | 0,50–0,78 giây/câu |
| DNS ngoài sau khi bật offline guard | Bị chặn như thiết kế |
| API/worker sau restart | PASS — cả hai `RUNNING`, `/v1/health` trả `ok` |
| Catalog API | PASS — cả E2B và 31B `installed=true`, `valid=true` |
| TranslationStage thật với 31B | PASS — 2 block, artifact SHA hợp lệ, kết thúc `READY_TTS` ở 650‰ |

Smoke corpus gồm Anh, Nhật, Thái, Hàn và Ả Rập sang tiếng Việt. Sáu yêu cầu,
bao gồm một câu mang nội dung prompt-injection, đều kết thúc bằng `finish_reason=stop`,
không thêm Markdown/lời giải thích và giữ đúng ý trong kiểm tra nghe/đọc thủ công.
Độ trễ từng yêu cầu trong script nghiệm thu lặp lại nằm trong khoảng
0,50–0,78 giây với câu ngắn.

## Kiểm thử tự động

- Windows: 217 pass, 1 skip do integration FFmpeg không có binary tại máy local.
- Ubuntu GPU VM: **218/218 pass**.
- 30 test riêng cho normalize/merge/split block, artifact, retry output rỗng,
  Vietnamese bypass, model selection, checkpoint/resume và offline fake backend.
- Test backend llama.cpp bao phủ command không qua shell, loopback-only, health,
  tokenizer, timeout, malformed output, prompt injection và terminate/kill.

## Hợp đồng vận hành

- Mỗi block tối đa 12 giây và 128 token theo tokenizer của đúng GGUF.
- Kết quả rỗng được chia đôi và retry đúng một lần; không copy nguyên văn nguồn.
- Chỉ commit block sau khi output hoàn chỉnh; restart bỏ qua block đã commit.
- Whisper và Gemma không được giữ đồng thời; `llama-server` được đóng ở cuối stage,
  khi hủy hoặc khi có lỗi.
- Docker worker dùng `network_mode: none`. VM native dùng Python audit hook chặn
  DNS/TCP ngoài và chỉ cho đúng `127.0.0.1:18081`; đây không phải network namespace.
- Bản 31B là mặc định vì người dùng yêu cầu model nhiều tham số nhất phù hợp GPU
  24 GiB. Không tự động fallback sang E2B nếu 31B lỗi hoặc thiếu VRAM.
