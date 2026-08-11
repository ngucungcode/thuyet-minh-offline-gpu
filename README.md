# Thuyết Minh Offline GPU

Thuyết Minh Offline GPU là hệ thống tự phục vụ để tạo bản thuyết minh tiếng Việt
trên máy chủ NVIDIA, vận hành qua dashboard web hoặc CLI `dub`.

Sau khi tải nguồn xong, ASR, dịch, tách thoại, TTS và dựng video đều chạy cục bộ.
Dự án không dùng suy luận đám mây, analytics hay telemetry.

> Chỉ tải và xử lý nội dung, phụ đề và giọng nói mà bạn sở hữu, được cấp phép hoặc
> có quyền sử dụng hợp pháp. Dự án không vượt DRM, không kèm indexer/tracker và
> không xác định quyền sử dụng thay cho người dùng.

## Tính năng

- Dashboard tiếng Việt cho tải MP4/MKV + SRT, tìm nguồn, chọn model, tạo và theo dõi job.
- Prowlarr và qBittorrent cho nguồn do người vận hành tự cấu hình hợp pháp.
- Ưu tiên phụ đề nhúng, sidecar hoặc OpenSubtitles; fallback ASR CUDA offline.
- Dịch sang tiếng Việt bằng Gemma 4 qua `llama.cpp` CUDA.
- TIGER-DnR loại thoại diễn viên nhưng giữ nhạc và hiệu ứng.
- VieNeu v2 hoặc Piper tạo lời thuyết minh tiếng Việt.
- Chế độ nhịp tự nhiên dịch gọn theo thời lượng, mượn khoảng lặng lân cận và giới hạn
  tốc độ toàn câu ở 1,20×. Nếu TTS đo thực tế vẫn dài, hệ thống tự dùng Gemma rút
  gọn đúng khối lỗi rồi chỉ tổng hợp lại khối đó; vẫn có chế độ khớp timestamp
  nghiêm ngặt khi cần.
- Ducking, mix và xuất MP4 H.264 + AAC: nguồn H.264 được passthrough; nguồn HEVC
  chỉ mã hóa lại phần hình sang H.264 bằng CPU để hoạt động nhất quán trên mọi GPU hỗ trợ.
- Checkpoint atomic, hủy, retry và tiếp tục sau khi tiến trình khởi động lại.
- Tiến độ realtime theo stage, tốc độ tải, ETA, segment và số block.
- Xuất MP4, phụ đề tiếng Việt SRT và timing report JSON.
- Một job GPU nặng tại một thời điểm để kiểm soát VRAM.

## Môi trường được hỗ trợ

Đường triển khai production native có các yêu cầu sau:

- Ubuntu 22.04 x86_64.
- Python 3.11 hoặc 3.12 tại lệnh `python3`.
- NVIDIA GPU thuộc ma trận kiến trúc CUDA bên dưới.
- Ít nhất 16 GiB RAM.
- CUDA toolkit 12.6 hoặc 12.8 có `nvcc` tại `/usr/local/cuda/bin/nvcc`.
- NVIDIA driver tối thiểu 560.28.03 với CUDA 12.6 hoặc 570.26 với CUDA 12.8;
  `nvidia-smi` phải nhìn thấy GPU.
- PyTorch CUDA đã được image nhà cung cấp cài sẵn và `torch.cuda.is_available()` trả `True`.
- Quyền `root` hoặc tài khoản có `sudo`.

Trình cài không thay NVIDIA driver, CUDA toolkit hoặc PyTorch của nhà cung cấp.
Docker Compose dành cho host có Docker daemon và NVIDIA Container Toolkit là
đường triển khai nâng cao.

### Ma trận GPU và kiến trúc CUDA

| CUDA target | GPU tiêu biểu | Trạng thái | Profile khuyến nghị theo VRAM |
|---|---|---|---|
| `sm_70` | Tesla V100 16/32 GiB | Hỗ trợ có giới hạn; maintenance-limited | `balanced` với 16 GiB, `maximum` với 32 GiB |
| `sm_75` | Tesla T4 16 GiB | Hỗ trợ | `balanced` |
| `sm_80` | NVIDIA A100 40/80 GiB, A30 24 GiB | Hỗ trợ | `maximum` |
| `sm_86` | NVIDIA A10 24 GiB, A40 48 GiB, GeForce RTX 3090 24 GiB | Hỗ trợ | `maximum` |
| `sm_89` | NVIDIA L4 24 GiB, L40/L40S 48 GiB | Hỗ trợ | `maximum` |
| `sm_90` | NVIDIA H100/H200 | Hỗ trợ | `maximum` |
| `sm_80` | NVIDIA CMP 170HX 8 GiB | Thử nghiệm; chỉ bật khi toàn bộ probe đạt | Chỉ `minimal` |

V100 là nhánh maintenance-limited: dự án chỉ duy trì tương thích và sửa lỗi theo khả
năng, không cam kết nâng toolchain vượt quá phiên bản CUDA/PyTorch còn hỗ trợ Volta.
CMP 170HX không được tự động nâng lên `balanced` hoặc `maximum`, kể cả khi driver báo
dung lượng khác. Installer xác minh driver, CUDA, kernel FP16 PyTorch, CTranslate2 và
binary native; trước khi dùng production vẫn phải chạy full acceptance trên chính card đó.

Pascal (`sm_60`, `sm_61`, `sm_62`) và kiến trúc cũ hơn không được hỗ trợ. Kiến trúc
không có trong bảng cũng bị từ chối theo nguyên tắc fail-closed cho đến khi có build và
báo cáo nghiệm thu riêng; PTX fallback không được xem là bằng chứng tương thích
production.

Image Docker release dùng CUDA fatbin chứa mã cho `sm_70`, `sm_75`, `sm_80`, `sm_86`,
`sm_89` và `sm_90`. Cài native chỉ build artifact cho compute capability của GPU đã
được preflight chọn trên máy đó để giảm thời gian build và kích thước binary; không sao
chép native binary giữa các máy có kiến trúc khác nhau.

Trên host nhiều GPU, mọi ngưỡng VRAM/profile đều áp dụng cho **CUDA logical device 0**,
không lấy card lớn nhất và không cộng VRAM. Installer native ghi UUID card đã chọn vào
`CUDA_VISIBLE_DEVICES` cùng kiến trúc build trong `.env.native`; worker từ chối khởi động
nếu UUID, kiến trúc hoặc CUDA toolkit không còn khớp sau reboot. Muốn chọn card native cụ thể, thêm
`--gpu-device HOST_INDEX_OR_GPU_UUID` vào lệnh installer; card được chọn trở thành logical
device 0 và được pin bằng UUID sau preflight. Với Docker Compose, đặt
`DUB_GPU_DEVICE_ID` trong `.env` thành host index hoặc UUID NVIDIA; Compose chỉ cấp đúng
card đó cho worker. Không dùng chung `.env.native` giữa hai máy.

Ma trận trên là hợp đồng build, không thay thế nghiệm thu phần cứng. Trước khi công bố
một GPU ở trạng thái production, release phải được chạy trên chính phần cứng đó và có
báo cáo tối thiểu cho preflight, native compile/load, suy luận từng stage, peak VRAM,
thời gian xử lý và kiểm tra file đầu ra.

### Profile model

| Profile | VRAM phù hợp | Dung lượng trống tối thiểu | Model chính |
|---|---:|---:|---|
| `auto` | Tự phát hiện | Theo profile được chọn | Tự chọn cấu hình bên dưới |
| `maximum` | ≥22 GiB | 55 GiB | Large-v3-Turbo, Gemma 4 31B, TIGER-DnR, VieNeu |
| `balanced` | ≥8 GiB | 35 GiB | Whisper Small, Gemma 4 E2B, TIGER-DnR, VieNeu |
| `minimal` | ≥6 GiB | 25 GiB | Whisper Small, Gemma 4 E2B, TIGER-DnR, Piper |
| `none` | Không cài model | 20 GiB | Chỉ runtime và dịch vụ |

Các mức trên là dung lượng trống trước khi cài. Hãy dự phòng thêm dung lượng cho
phim nguồn, artifact trung gian, output và backup nâng cấp. Profile được chọn theo VRAM
của một GPU đã qua preflight, không cộng VRAM của nhiều GPU. `auto` chọn `maximum` khi
có ít nhất 22 GiB, chọn `balanced` khi có từ 8 GiB đến dưới 22 GiB, chọn `minimal` khi có
từ 6 GiB đến dưới 8 GiB và từ chối cài model nếu thấp hơn 6 GiB. Người vận hành có thể chủ
động chọn `minimal` để giảm thời gian cài
và mức dùng tài nguyên; riêng CMP 170HX luôn bị giới hạn ở profile này.

Installer native với `--profile auto` ghi lựa chọn model theo VRAM thực tế vào
`.env.native`. Docker Compose không có bước chọn profile đó nên mặc định fail-safe với
`minimal` (Whisper Small, Gemma 4 E2B, TIGER-DnR và Piper), phù hợp mọi card trong
allowlist có từ 6 GiB VRAM. Máy nhiều VRAM có thể đặt bốn biến
`DUB_DEFAULT_ASR_MODEL_ID`, `DUB_DEFAULT_TRANSLATION_MODEL_ID`,
`DUB_DEFAULT_SEPARATION_MODEL_ID` và `DUB_DEFAULT_TTS_MODEL_ID` trong `.env` theo
profile đã cài; worker vẫn kiểm tra `minimum_vram_mib` ngay trước stage dùng model.

## Cài nhanh

Trên Ubuntu đáp ứng đủ yêu cầu, chạy:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/latest/download/install.sh | sudo bash
```

Installer kiểm tra toàn bộ preflight trước khi sửa hệ thống, tự chọn profile,
tải và xác minh SHA-256 model, tạo secret, khởi động stack, chạy acceptance cơ
bản và cài `dub` tại `/usr/local/bin/dub`.

Bootstrap mặc định dùng bộ smoke check production, tận dụng pip cache bên trong
data root và tự dùng tối đa 16 luồng CPU khi build native. Để chạy thêm toàn bộ
unit test ngay trên máy cài đặt, dùng chế độ chậm `full`:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/latest/download/install.sh | sudo env DUB_INSTALL_TEST_MODE=full bash
```

Có thể giới hạn số luồng compile bằng `DUB_BUILD_JOBS`; giá trị phải là số
nguyên dương. Cả hai biến chỉ ảnh hưởng cold bootstrap, không thay đổi model
hoặc checksum của artifact đã cài.

Installer ghi số giây của bootstrap, cài model, acceptance và toàn bộ lượt chạy
vào `install-state.json` dưới khóa `performance`. Trên model mount Linux chỉ-đọc,
worker vẫn băm SHA-256 đầy đủ trước lần dùng đầu tiên; các stage/job tiếp theo chỉ
dùng fast-path khi danh sách file và metadata không đổi. Thư mục model có thể ghi
luôn được băm đầy đủ ở mỗi lần dùng. Cache mất khi worker khởi động lại. Tokenizer
dịch cũng dùng LRU hữu hạn, còn tiến
độ TTS/timing được gộp theo phần nghìn nhưng checkpoint từng block vẫn được ghi.

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

### Tải file có sẵn — cách nhanh nhất

Trong dashboard, chọn **Tải file lên**, rồi:

1. Chọn một file video `.mp4` hoặc `.mkv` trên máy của bạn.
   Luồng hình chính H.264/AVC được passthrough nhanh, còn HEVC/H.265 SDR được mã hóa
   lại riêng phần hình sang H.264/AVC bằng `libx264`; vì vậy nguồn HEVC sẽ xuất chậm hơn.
   HEVC HDR10, HLG và Dolby Vision bị từ chối cho đến khi có tone-map được kiểm thử,
   tránh tạo video tối hoặc sai màu.
   VP8, FFV1 và MPEG-4 Part 2 vẫn bị từ chối ngay khi finalize thay vì đợi đến công
   đoạn xuất. Cover-art không được xem là luồng hình chính.
2. Có thể chọn thêm file phụ đề `.srt`; nếu có SRT, hãy chọn đúng ngôn ngữ nguồn.
3. Giữ **Nhịp tự nhiên** để ưu tiên giọng đều, hoặc chọn **Khớp nghiêm ngặt** nếu
   timestamp gốc quan trọng hơn độ tự nhiên.
4. Chọn model/giọng, xác nhận quyền sử dụng và bắt đầu tải lên.
5. Không đóng trang trong lúc upload; khi server đã nhận đủ file, job có checkpoint
   và có thể tiếp tục độc lập trên máy chủ.

Nếu finalize tạm bị chặn bởi một job khác, SRT chưa hợp lệ hoặc kết nối gián đoạn,
dashboard giữ mã session và lần bấm **Bắt đầu** kế tiếp chỉ gửi artifact còn thiếu;
video/SRT đã nhận đủ và khớp SHA-256 không bị tải lại. Session chỉ bị xóa khi người dùng chủ động hủy,
bấm **Xóa phiên tạm**, hoặc khi server dọn session chưa finalize đã hết TTL. TTL mặc
định là 7 ngày (`604800` giây), cấu hình bằng `DUB_UPLOAD_SESSION_TTL_SECONDS` và
không được vượt quá 90 ngày.

CLI tương đương:

```bash
dub upload ./phim.mkv --subtitle ./phim.en.srt \
  --source-language en --timing-profile natural \
  --i-have-rights --wait -o result.mp4
```

Nếu không có SRT, bỏ `--subtitle`; pipeline sẽ chạy ASR offline từ âm thanh video.
File được truyền theo luồng và ghi vào `.part` trước khi đổi tên atomic, nên API không
nạp cả phim vào RAM. Giới hạn mặc định là 100 GiB cho video và 16 MiB cho SRT;
session chưa hoàn tất được kiểm tra định kỳ và tự dọn theo TTL cấu hình bằng
`DUB_UPLOAD_SESSION_TTL_SECONDS` (mặc định 7 ngày).

### Tìm nguồn qua Prowlarr

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
  --subtitle-mode prefer --timing-profile natural --wait -o result.mp4
```

`RELEASE_ID` phải lấy từ một kết quả tìm kiếm hợp lệ. V1 chỉ nhận loại nội dung
`movie`; hệ thống không nhận URL tải từ xa tùy ý. File cục bộ phải đi qua luồng upload
MP4/MKV có kiểm tra định dạng ở trên.

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

Nguồn của job đã finalize trong `incoming` nằm ngoài lệnh maintenance cleanup;
chỉ session upload chưa finalize đã quá TTL (mặc định 7 ngày, cấu hình bằng
`DUB_UPLOAD_SESSION_TTL_SECONDS`) được tự dọn. Luôn đọc dry-run trước khi dùng
`--apply`.

## Nâng cấp, cài bản ghim và rollback

Để nâng cấp deployment Git sạch từ `v0.2.0` đến `v0.3.5` lên `v0.3.6`, chạy
một lệnh:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.3.6/install.sh | sudo bash -s -- --upgrade-existing --yes
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
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.3.6/install.sh | sudo bash -s -- --upgrade-existing --yes
```

Xóa file khi khóa còn được giữ sẽ tạo inode mới và có thể cho phép hai installer chạy song
song. Từ `v0.2.4`, installer đóng descriptor khóa trước khi tạo daemon và cho phép source
mới thay script điều khiển stack mà không coi đó là thay đổi runtime ML; thao tác dừng
stack này chỉ cần thực hiện một lần khi nâng cấp từ deployment bị ảnh hưởng.

Trình cài chỉ chấp nhận đường nâng cấp đã khai báo, từ chối worktree bẩn, origin sai
hoặc còn job đang hoạt động. Source mới được kích hoạt bằng transaction có journal;
model, virtualenv, dữ liệu, checkpoint và cấu hình tùy chỉnh của quản trị viên được
giữ nguyên. Trình cài chỉ cập nhật nguyên tử các khóa GPU do nó quản lý và chuyển
những model mặc định legacy còn nguyên giá trị cũ sang mặc định an toàn của profile;
giá trị model đã tùy chỉnh không bị thay đổi. Nếu health check hoặc acceptance thất
bại, trình cài phục hồi source và trạng thái stack cũ. Backup source cũ được giữ lại
để kiểm tra thủ công.

Job tạo trước `v0.3.0` không có `timing_profile` được giữ ở chế độ `strict`, vì vậy
nâng cấp không đổi timestamp hay tái tạo TTS giữa chừng. Job mới mặc định dùng
`natural`. Khi nâng lên `v0.3.6`, job cũ dừng ở lỗi `timing_rewrite_required`
hoặc `timing_rewrite_exhausted` được chuyển thành có thể tiếp tục. Cơ chế adaptive
dùng chính bản tiếng Việt trước đó cùng thời lượng TTS đã đo trong checkpoint để đặt
ngân sách từ cứng theo cửa sổ còn lại. Stage và các block đã hoàn tất được giữ nguyên;
hệ thống chỉ rút gọn rồi tổng hợp TTS lại block bị tràn. Chạy:

```bash
dub resume JOB_ID
```

Để cài mới đúng bản `v0.3.6` thay vì `latest`, dùng URL ghim theo tag:

```bash
set -o pipefail; curl -fsSL https://github.com/ngucungcode/thuyet-minh-offline-gpu/releases/download/v0.3.6/install.sh | sudo bash
```

Installer có thể chạy lại an toàn trên đúng commit đã cài: không reset worktree có
thay đổi và không ghi đè các giá trị tùy chỉnh trong `.env.native`; liên kết GPU được
kiểm tra lại rồi lưu nguyên tử. Nếu commit đích khác mà không có
`--upgrade-existing`, installer dừng trước khi checkout hoặc sửa runtime.

Deployment legacy không có Git cũng bị từ chối theo mặc định. Cờ
`--migrate-existing` chỉ dành cho bản source có runtime fingerprint giống hệt;
khác fingerprint sẽ dừng trước khi đổi source. Không dùng cờ này để ép nâng
`v0.1.x` lên `v0.3.x`.

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

GPU phải thuộc allowlist `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89` hoặc `sm_90`,
đồng thời driver, CUDA toolkit và PyTorch phải tương thích với nhau. Pascal và cũ hơn
không được hỗ trợ. Với V100, dùng toolchain còn hỗ trợ Volta; với CMP 170HX, chỉ dùng
profile `minimal` sau khi toàn bộ probe thử nghiệm đạt. Việc `nvidia-smi` nhìn thấy card
không thay thế smoke test native và nghiệm thu trên phần cứng thật.

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
- [SBOM CycloneDX 1.6 của release](release/sbom.cdx.json); SBOM tạo trên máy đã cài còn
  xác minh build receipt và ghi CUDA/binary `llama.cpp` thực tế.
- [Thông báo dependency và model](THIRD_PARTY_NOTICES.md)
- [Báo lỗi trên GitHub](https://github.com/ngucungcode/thuyet-minh-offline-gpu/issues)

Mã nguồn được phát hành theo [GPL-3.0-or-later](LICENSE). Việc dùng phần mềm không
trao quyền đối với phim, phụ đề, torrent, model giọng hoặc giọng nói của bên thứ ba.
