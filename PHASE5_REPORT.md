# Phase 5 — Độ bền, vận hành và hoàn thiện sản phẩm

Ngày nghiệm thu: 01/08/2026<br>
Máy production: NVIDIA GeForce RTX 3090 24 GiB, driver 610.43.02

## Kết luận

**PASS cổng triển khai native GPU.** API, worker, Prowlarr và qBittorrent đang
`RUNNING`; `/v1/health` trả `status=ok`, `gpu.ready=true`, catalog có 9 model và
worker heartbeat hoạt động. Bản trước triển khai được lưu tại:

- `/workspace/thuyet-minh-backups/pre-phase45-20260801T152801Z.tar.gz`
- `/workspace/thuyet-minh-backups/pre-hardening-20260801T161509Z.tar.gz`

## Kiểm thử cuối

| Môi trường | Kết quả |
|---|---|
| Windows local | **302 passed, 2 skipped**, 1 warning deprecation |
| Ubuntu RTX 3090 | **304 passed**, 3 warning deprecation |
| VieNeu native smoke | Hai lượt persistent PASS, WAV 2,077 s và 1,764 s |
| Piper native acceptance | PASS, H.264 + AAC, sync 0 µs |
| VieNeu 10 s / 5 phút / 30 phút | PASS cả ba, RTF lần lượt 3,555 / 0,557 / 0,493 |

Các warning đến từ API deprecation của Starlette/AnyIO, không phải lỗi pipeline.

## Chức năng hoàn thiện

- Tiến độ chi tiết theo stage qua SQLite event và SSE, gồm progress separation,
  TTS block, timing và thời gian FFmpeg export.
- Một job GPU tại một thời điểm; cancel/resume, checkpoint block và artifact.
- CLI/API chọn model separation/TTS; cài và verify model chỉ qua model-manager.
- `dub fetch` tải video/subtitle/timing atomically; server kiểm SHA trước khi gửi.
- VieNeu chất lượng cao mặc định; Piper CPU là lựa chọn nhẹ đã nghiệm thu thật.
- Cleanup mặc định dry-run và không chạm `incoming`.

## Hardening bổ sung trong cổng cuối

- Process group mới bảo đảm hủy cả runtime Python và FFmpeg con trên POSIX;
  Windows có process group và `taskkill /T /F` khi buộc dừng.
- Export checkpoint được tái sử dụng sau process death chỉ khi MP4 và toàn bộ
  input binding còn đúng; tránh encode lại stage đã hoàn tất.
- Download artifact từ chối path escape, symlink, file không regular, SHA/size
  sai hoặc file bị sửa sau khi COMPLETED.
- Cleanup `--apply` gom action theo job và giữ `BEGIN IMMEDIATE` trong lúc
  revalidate revision/status/retention rồi xóa. Resume không thể chen vào giữa
  snapshot và deletion.

## Offline và chuỗi cung ứng

- Acceptance chặn thành công DNS, IPv4 và IPv6 bằng native offline audit guard;
  các biến Hugging Face/Transformers/W&B đều ở chế độ offline.
- Docker worker vẫn dùng `network_mode: none`; native VM dùng audit guard, không
  tuyên bố tương đương network namespace ở tầng kernel.
- CycloneDX 1.6 SBOM cuối có **123 component** tại
  `var/reports/sbom.cdx.json`, SHA-256:
  `30de17160d30cdfec15623a6ea4d8d8831eb2bd44579d3223c526b469ba43072`.
- `models.lock.json`, `components.lock.json`, GPL license và
  `THIRD_PARTY_NOTICES.md` được giữ trong bản triển khai.

## Retention cleanup

Dry-run cuối trả `action_count=0`, `errors=[]`; không file nào bị xóa. Lệnh thực
thi vẫn yêu cầu cờ rõ ràng:

```bash
.venv-native/bin/python scripts/cleanup-job-artifacts.py
.venv-native/bin/python scripts/cleanup-job-artifacts.py --apply
```

`--apply` chỉ nên chạy sau khi đọc JSON plan. Job đang chạy, COMPLETED, lỗi không
retry, lỗi retry chưa đủ 7 ngày và mọi source trong `incoming` đều ngoài phạm vi.

## Giới hạn vận hành còn lại

- Fixture kỹ thuật không thay thế kiểm nghe phim có quyền sử dụng; cần corpus
  thực để chấm mức rò thoại, tên riêng và độ tự nhiên của thuyết minh.
- API/WebUI chỉ bind loopback. Trước khi công bố ra LAN/Internet phải bổ sung
  authentication, TLS và firewall; hiện chưa được phép expose trực tiếp.
- Native audit hook chặn Python socket nhưng không phải kernel namespace; môi
  trường yêu cầu cô lập mạnh nên dùng Docker worker `network_mode: none` hoặc
  firewall outbound của VM.

## Artifact nghiệm thu

Các JSON và MP4 nghiệm thu nằm dưới `var/state/phase4-acceptance-artifacts` trên
máy production. Bản sao JSON/SBOM đã được kéo về `.artifacts/remote-results` của
workspace để đối chiếu độc lập.
