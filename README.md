# Thuyết Minh Offline GPU

Thuyết Minh Offline GPU là hệ thống tự phục vụ để tạo bản thuyết minh tiếng Việt
trên máy chủ NVIDIA, vận hành qua dashboard web hoặc CLI `dub`.

Sau khi tải nguồn xong, ASR, dịch, tách thoại, TTS và dựng video đều chạy cục bộ.
Dự án không dùng suy luận đám mây, analytics hay telemetry.

> Chỉ tải và xử lý nội dung, phụ đề và giọng nói mà bạn sở hữu, được cấp phép hoặc
> có quyền sử dụng hợp pháp. Dự án không vượt DRM, không kèm indexer/tracker và
> không xác định quyền sử dụng thay cho người dùng.

## Tính năng

- Dashboard tiếng Việt cho tìm nguồn, chọn model, tạo và theo dõi job.
- Prowlarr và qBittorrent cho nguồn do người vận hành tự cấu hình hợp pháp.
- Ưu tiên phụ đề nhúng, sidecar hoặc OpenSubtitles; fallback ASR CUDA offline.
- Dịch sang tiếng Việt bằng Gemma 4 qua `llama.cpp` CUDA.
- TIGER-DnR loại thoại diễn viên nhưng giữ nhạc và hiệu ứng.
- VieNeu v2 hoặc Piper tạo lời thuyết minh tiếng Việt.
- Khớp từng slot thoại, ducking, mix và xuất H.264 passthrough + AAC.
- Checkpoint atomic, hủy, retry và tiếp tục sau khi tiến trình khởi động lại.
- Tiến độ realtime theo stage, tốc độ tải, ETA, segment và số block.
- Xuất MP4, phụ đề tiếng Việt SRT và timing report JSON.
- Một job GPU nặng tại một thời điểm để kiểm soát VRAM.

## Môi trường được hỗ trợ

Trình cài production `v0.2.3` hỗ trợ đường triển khai native sau:

- Ubuntu 22.04 x86_64.
- Python 3.11 hoặc 3.12 tại lệnh `python3`.
- NVIDIA GPU có compute capability chính xác `8.6` (`sm_86`).
- Ít nhất 16 GiB RAM.
- NVIDIA driver hoạt động và `nvidia-smi` nhìn thấy GPU.
- CUDA toolkit có `nvcc` tại `/usr/local/cuda/bin/nvcc`.
- PyTorch CUDA đã được image nhà cung cấp cài sẵn và `torch.cuda.is_available()` trả `True`.
- Quyền `root` hoặc tài khoản có `sudo`.

Trình cài không thay NVIDIA driver, CUDA toolkit hoặc PyTorch của nhà cung cấp.
Docker Compose dành cho host có Docker daemon và NVIDIA Container Toolkit là
đường triển khai nâng cao; image hiện tại cũng được build riêng cho `sm_86`.

### Profile model

| Profile | VRAM phù hợp | Dung lượng trống tối thiểu | Model chính |
|---|---:|---:|---|
| `auto` | Tự phát hiện | Theo profile được chọn | Tự chọn cấu hình bên dưới |
| `maximum` | ≥22 GiB | 55 GiB | Large-v3-Turbo, Gemma 4 31B, TIGER-DnR, VieNeu |
| `balanced` | ≥8 GiB | 35 GiB | Whisper Small, Gemma 4 E2B, TIGER-DnR, VieNeu |
| `minimal` | ≥6 GiB | 25 GiB | Whisper Small, Gemma 4 E2B, TIGER-DnR, Piper |
| `none` | Không cài model | 20 GiB | Chỉ runtime và dịch vụ |

Các mức trên là dung lượng trống trước khi cài. Hãy dự phòng thêm dung lượng cho
phim nguồn, artifact trung gian, output và backup nâng cấp.

## Cài nhanh

Trên Ubuntu đáp ứng đủ yêu cầu, chạy:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/latest/download/install.sh | sudo bash
```

Installer kiểm tra toàn bộ preflight trước khi sửa hệ thống, tự chọn profile,
tải và xác minh SHA-256 model, tạo secret, khởi động stack, chạy acceptance cơ
bản và cài `dub` tại `/usr/local/bin/dub`.

Kiểm tra sau cài:

```bash
dub doctor
dub stack status
dub health
```

Nếu đang ở shell `root` trên GPU container không có `sudo`, thay phần cuối lệnh
cài bằng `| bash`.

### Mở dashboard

Các giao diện quản trị chỉ bind loopback. Từ máy cá nhân, mở một SSH tunnel bằng
lệnh một dòng sau; thay `SSH_PORT` và `GPU_HOST` bằng thông tin máy chủ:

```bash
ssh -p SSH_PORT -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 -L 9696:127.0.0.1:9696 root@GPU_HOST
```

Giữ phiên SSH này mở, sau đó truy cập:

| Dịch vụ | Địa chỉ trên máy cá nhân |
|---|---|
| Dashboard và API | `http://127.0.0.1:8080/` |
| OpenAPI | `http://127.0.0.1:8080/docs` |
| qBittorrent WebUI | `http://127.0.0.1:8081/` |
| Prowlarr | `http://127.0.0.1:9696/` |

## Tạo bản thuyết minh đầu tiên

1. Mở Prowlarr và thêm indexer mà bạn được phép sử dụng; dự án không cài sẵn indexer.
2. Trong dashboard, kiểm thử Prowlarr/qBittorrent và đăng nhập OpenSubtitles nếu cần.
3. Tìm tên phim và năm, rồi chọn đúng release từ kết quả.
4. Chọn ngôn ngữ nguồn, chế độ phụ đề và các model đã cài/verify.
5. Xác nhận quyền sử dụng nội dung và giọng tham chiếu, nếu có, rồi tạo job.
6. Nếu job yêu cầu chọn phụ đề hoặc ngôn ngữ, xử lý ngay trong thẻ chi tiết.
7. Theo dõi tiến độ realtime và tải MP4/SRT/timing report khi hoàn tất.

CLI tương đương:

```bash
dub search "Tên phim" --year 2024
dub submit --release-id RELEASE_ID --i-have-rights \
  --subtitle-mode prefer --wait -o result.mp4
```

`RELEASE_ID` phải lấy từ một kết quả tìm kiếm hợp lệ. V1 chỉ nhận loại nội dung
`movie`; hệ thống không nhận URL tải tùy ý từ client.

## Vận hành

```bash
dub stack status
dub stack logs --lines 200
dub jobs list --limit 20
dub status JOB_ID
dub watch JOB_ID --fetch-dir ./results
dub cancel JOB_ID
dub resume JOB_ID
dub fetch JOB_ID --kind all -o ./results
```

Quản lý model:

```bash
dub models profiles
dub models recommend
dub models list
dub models install-profile maximum --yes
```

Cleanup mặc định chỉ in kế hoạch và không xóa file:

```bash
dub maintenance cleanup
dub maintenance cleanup --apply --yes
dub maintenance sbom
```

Job đã hoàn tất và toàn bộ thư mục nguồn `incoming` nằm ngoài phạm vi cleanup tự
động. Luôn đọc dry-run trước khi dùng `--apply`.

## Nâng cấp, cài bản ghim và rollback

Để nâng cấp deployment Git sạch từ `v0.2.0`, `v0.2.1` hoặc `v0.2.2` lên `v0.2.3`, chạy
một lệnh:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.2.3/install.sh | sudo bash -s -- --upgrade-existing --yes
```

Deployment `provider` được cài bởi release cũ có thể để `supervisord` kế thừa khóa
installer. Nếu gặp `Một trình cài khác đang chạy`, không xóa file lock. Trước tiên kiểm tra
tiến trình giữ khóa; nếu không còn installer/model bootstrap thật và chỉ stack cũ giữ khóa,
dừng stack rồi thử lại:

```bash
LOCK=/run/lock/thuyet-minh-offline-install.lock
sudo lslocks -o PID,COMMAND,TYPE,MODE,PATH | grep -F "$LOCK" || true
pgrep -af 'install\.sh|bash -s|native-bootstrap|native-model|apt-get|supervisord' || true
dub jobs list --limit 20
# Chỉ dừng stack khi danh sách trên không còn job đang xử lý.
dub stack stop
sudo flock -n "$LOCK" true && echo LOCK_FREE
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.2.3/install.sh | sudo bash -s -- --upgrade-existing --yes
```

Xóa file khi khóa còn được giữ sẽ tạo inode mới và có thể cho phép hai installer chạy song
song. `v0.2.3` đóng descriptor khóa trước khi tạo daemon nên thao tác dừng stack này chỉ cần
thực hiện một lần khi nâng cấp từ deployment bị ảnh hưởng.

Trình cài chỉ chấp nhận đường nâng cấp đã khai báo, từ chối worktree bẩn, origin sai
hoặc còn job đang hoạt động. Source mới được kích hoạt bằng transaction có journal;
`.env.native`, model, virtualenv và dữ liệu được giữ nguyên. Nếu health check hoặc
acceptance thất bại, trình cài phục hồi source và trạng thái stack cũ. Backup source
cũ được giữ lại để kiểm tra thủ công.

Sau khi nâng cấp bản vá này, job từng dừng ở lỗi `output_track_layout_invalid` hoặc
`output_duration_mismatch` sẽ có thể tiếp tục từ checkpoint dựng MP4. Bản xuất mới
lấy điểm kết thúc luồng hình làm timeline chuẩn, đệm/cắt phần tiếng tương ứng mà
không mã hóa lại hoặc cắt luồng hình:

```bash
dub resume JOB_ID
```

Để cài mới đúng bản `v0.2.3` thay vì `latest`, dùng URL ghim theo tag:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.2.3/install.sh | sudo bash
```

Installer có thể chạy lại an toàn trên đúng commit đã cài: không reset worktree có
thay đổi và không ghi đè `.env.native`. Nếu commit đích khác mà không có
`--upgrade-existing`, installer dừng trước khi checkout hoặc sửa runtime.

Deployment legacy không có Git cũng bị từ chối theo mặc định. Cờ
`--migrate-existing` chỉ dành cho bản source có runtime fingerprint giống hệt;
khác fingerprint sẽ dừng trước khi đổi source. Không dùng cờ này để ép nâng
`v0.1.x` lên `v0.2.x`.

Trước mọi thay đổi release, hãy snapshot toàn bộ thư mục cài và data volume. Không
chỉ đổi source về bản cũ trong khi giữ database từ release mới hơn. Giữ nguyên
backup và journal nếu installer yêu cầu phục hồi thủ công.

## Bảo mật và mạng

- Chỉ cần mở cổng SSH của máy chủ để quản trị qua tunnel.
- Không public trực tiếp các cổng `8080`, `8081` hoặc `9696`; chúng không có TLS công khai.
- Cổng peer theo cấu hình qBittorrent là tùy chọn; Docker mặc định dùng TCP/UDP
  `6881`. Chỉ mở cổng peer thực tế qua firewall khi cần.
- Secret native nằm dưới `var/secrets` với mode `0600`; không đưa chúng vào Git hoặc log.
- API acquisition có mạng để tìm/tải nguồn; worker suy luận chỉ dùng model local đã verify.
- Docker worker dùng `network_mode: none`; native worker dùng offline audit guard ở tầng tiến trình.
- Model chỉ được tải bởi model-manager có egress; worker không tải model hoặc fallback cloud.
- OpenSubtitles là tùy chọn; API route chỉ chấp nhận host chính thức đã allowlist.
- Dashboard/API hiện được thiết kế cho một người vận hành tin cậy qua loopback/SSH.

## Khắc phục sự cố

- **`dub: command not found`:** installer chưa hoàn tất. Xem lỗi đầu tiên rồi chạy
  lại lệnh cài. Không chạy riêng `sudo bash -s -- ...` vì nó không tải installer.
- **Preflight GPU thất bại:** chạy ba lệnh kiểm tra sau:

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

GPU phải là `sm_86`; runtime hiện không hỗ trợ chung mọi GPU CUDA.

- **Stack hoặc dashboard không phản hồi:** chạy `dub stack status`, xem
  `dub stack logs --lines 200`, rồi dùng `dub stack restart`. Tạo lại tunnel và
  kiểm tra cổng local nếu stack đã khỏe.
- **Tìm kiếm không có kết quả:** thêm một indexer hợp pháp trong Prowlarr, bấm
  **Test**, **Save**, rồi kiểm thử lại tích hợp trong dashboard.
- **OpenSubtitles lỗi hoặc token hết hạn:** đăng nhập lại ở mục **Tích hợp**, hoặc
  chọn ASR để tiếp tục mà không dùng OpenSubtitles.
- **Job lỗi có thể retry:** đọc lỗi bằng `dub status JOB_ID`, sửa nguyên nhân rồi
  chạy `dub resume JOB_ID`; checkpoint hợp lệ không chạy lại stage đã hoàn tất.

## Tài liệu và giấy phép

- [Workflow web và cấu hình tích hợp](docs/WEB_WORKFLOW.md)
- [Catalog model bất biến](config/models.lock.json)
- [SBOM CycloneDX 1.6 của release](release/sbom.cdx.json)
- [Thông báo dependency và model](THIRD_PARTY_NOTICES.md)
- [Báo lỗi trên GitHub](https://github.com/ngucungcode/thuyet-minh-offline-gpu/issues)

Mã nguồn được phát hành theo [GPL-3.0-or-later](LICENSE). Việc dùng phần mềm không
trao quyền đối với phim, phụ đề, torrent, model giọng hoặc giọng nói của bên thứ ba.
