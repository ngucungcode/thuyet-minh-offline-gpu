# Báo cáo Phase 1 — GPU runtime, tải nguồn và phụ đề

Ngày nghiệm thu native: 31/07/2026<br>
Đường dẫn triển khai: `/workspace/thuyet-minh-offline`

## Kết luận

Phase 1 đã chạy trực tiếp trên GPU container thuê, không phụ thuộc Docker daemon hay
systemd. Cổng runtime, GPU, API, database và adapter acquisition đều đạt. Prowlarr
hiện chưa có indexer nên tìm nguồn thật trả danh sách rỗng; đây là cấu hình bên ngoài
do người dùng phải tự thêm cho nội dung mình có quyền tải, không phải lỗi runtime.

ASR, dịch, TTS và mux chưa được triển khai trong Phase 1. Catalog model vẫn là danh
sách ứng viên với revision/hash giữ chỗ; không được tuyên bố pipeline thuyết minh đã
hoàn tất chỉ vì GPU worker đang chạy.

## Môi trường đã đo

| Hạng mục | Kết quả |
|---|---|
| Hệ điều hành | Ubuntu 22.04.4 LTS, x86_64, managed container |
| GPU | NVIDIA GeForce RTX 3090, 24.576 MiB VRAM, compute capability 8.6 |
| Driver | 580.82.07 |
| RAM | 40 GiB |
| Dung lượng | khoảng 150 GiB trống tại thời điểm khảo sát |
| Python | 3.11.0rc1 của nhà cung cấp |
| PyTorch | 2.7.0+cu126, CUDA 12.6 |
| CTranslate2 | 4.8.1; có `float16`, `int8_float16` và `bfloat16` |
| qBittorrent | 4.4.1-2 từ Ubuntu Jammy |
| Prowlarr | 2.5.2.5491, tar x64 kiểm SHA-256 |
| FFmpeg | 4.4.2 từ Ubuntu Jammy |

## Kết quả kiểm thử

- Python compile: PASS.
- Pytest: **86 passed**.
- CUDA matrix multiplication PyTorch: PASS, checksum 256.
- CTranslate2 CUDA compute-type probe: PASS.
- API health: `status=ok`, `gpu.ready=true`, `acquisition_configured=true`.
- SQLite: PASS, journal mode `wal`.
- qBittorrent adapter thực: thêm magnet giả không có nội dung, negotiate API v4,
  resume, pause, đọc trạng thái và xóa task: PASS.
- Prowlarr adapter thực: API xác thực thành công; search không indexer trả `[]`: PASS.
- Supervisor: SIGTERM API và worker, cả hai tự khởi động lại với PID mới: PASS.
- Restart toàn stack: dừng sạch rồi bốn process trở lại `RUNNING`: PASS.
- Port quản trị `8080`, `8081`, `9696`: chỉ bind `127.0.0.1`: PASS.
- Secret Prowlarr/qBittorrent: owner `dub:dub`, mode `0600`: PASS.

Lệnh acceptance có thể chạy lại:

```bash
cd /workspace/thuyet-minh-offline
./scripts/native-acceptance.sh
```

## Kiến trúc native đã chốt

- Supervisor chạy bốn tiến trình foreground: qBittorrent → Prowlarr → GPU worker →
  một Uvicorn API process.
- Dữ liệu bền vững nằm trong `var/`: SQLite/heartbeat, model, incoming, jobs,
  output, cấu hình service, log, socket và secret.
- API gọi Prowlarr/qBittorrent qua loopback. qBittorrent dùng cùng đường dẫn tuyệt
  đối `var/data/incoming` với coordinator, không cần volume mapping.
- Adapter qBittorrent đọc `/api/v2/app/version` một lần: v4 dùng
  `pause`/`resume`, v5 dùng `stop`/`start`.
- PyTorch hệ thống của nhà cung cấp không bị pip ghi đè. Venv native dùng
  `--system-site-packages`; extra `managed-gpu` chỉ cài CTranslate2 và
  faster-whisper.
- Prowlarr binary được pin/verify và để root sở hữu; data/config do user `dub` sở
  hữu. Không bundle indexer hay tracker.

## Giới hạn còn lại

1. Managed container không có `CAP_SYS_ADMIN`/`CAP_NET_ADMIN`; không thể tạo Docker
   hoặc network namespace để chặn egress worker ở cấp kernel. Phase 1 worker chỉ
   heartbeat và không có network client. Các phase inference sau phải thêm offline
   guard và traffic acceptance riêng.
2. Khi provider restart toàn bộ container, không có systemd để tự bật stack. Cần
   chạy `scripts/native-stack.sh start` hoặc cấu hình provider startup command là
   `scripts/native-stack.sh foreground`.
3. qBittorrent 4.4.1 là bản distro cũ nhưng được giữ loopback và adapter đã negotiate
   đúng API. Khi đổi base image nên nâng lên qBittorrent 5 và chạy lại smoke hai nhánh.
4. Chưa có indexer hợp pháp và chưa đưa một torrent nội dung thật qua trạng thái
   `ready_offline`. Người dùng phải cấu hình indexer/content source được phép trước
   acceptance nguồn thật.
5. OpenSubtitles chưa có API key/token; hệ thống sẽ chỉ dùng embedded/sidecar rồi
   fallback ASR khi Phase 2 có ASR.
6. Model trong `config/models.lock.json` chưa pin revision/hash thật, chưa tải và
   chưa license-review cuối. Phase 2 không được tự động tải model trong worker.

## Điều kiện bắt đầu Phase 2

Runtime GPU không còn bị chặn. Trước khi chạy một corpus thật, cần chọn và pin model
ASR đầu tiên, cập nhật SHA-256/license, cài model bằng luồng model-manager có egress,
sau đó mount/dùng read-only trong worker. Nếu cần nghiệm thu acquisition end-to-end,
người dùng phải thêm một indexer và cung cấp một torrent thử mà họ có quyền sử dụng.
