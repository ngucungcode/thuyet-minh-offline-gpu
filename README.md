# Thuyết Minh Offline GPU

Biến thể Ubuntu/Docker của hệ thống thuyết minh video, dành cho nội dung mà
người dùng sở hữu hoặc được phép tải và xử lý. ASR, dịch và TTS chạy cục bộ;
không có suy luận đám mây. Dự án không cấu hình sẵn indexer/tracker, không vượt
DRM và không tự xác định quyền sử dụng thay cho người dùng.

## Cài một lệnh

Đường triển khai được hỗ trợ đầy đủ là Ubuntu 22.04 x86_64, Python 3.11/3.12,
NVIDIA CUDA compute capability 8.6, RAM từ 16 GiB và image nhà cung cấp đã có
PyTorch CUDA. Trình cài không thay driver/CUDA, kiểm tra toàn bộ điều kiện trước
khi sửa hệ thống, tự chọn model theo VRAM, khóa WebUI vào loopback, tạo secret,
chạy test/acceptance và cài lệnh `dub` toàn cục.

```bash
curl -fsSL https://raw.githubusercontent.com/ngucungcode/thuyet-minh-offline-gpu/v0.1.2/install.sh \
  | sudo bash -s -- --ref v0.1.2 --profile auto --start --yes
```

RTX 3090 24 GiB tự chọn profile `maximum` gồm Faster-Whisper Large-v3-Turbo,
Gemma 4 31B Q4, TIGER-DnR và VieNeu v2. Có thể xem trước hoàn toàn không ghi:

```bash
curl -fsSL https://raw.githubusercontent.com/ngucungcode/thuyet-minh-offline-gpu/v0.1.2/install.sh \
  | sudo bash -s -- --ref v0.1.2 --dry-run
```

Sau khi cài:

```bash
dub doctor
dub stack status
dub models profiles
dub search "Tên phim" --year 2024
dub submit --release-id RELEASE_ID --i-have-rights --subtitle-mode asr \
  --wait --output result.mp4
```

`install.sh --help` liệt kê profile `maximum|balanced|minimal|none`, thư mục cài,
data volume, Git ref, autostart và cổng nghiệm thu. Trình cài idempotent: không
reset worktree có thay đổi, không ghi đè `.env.native`, không xoay secret sai
ngữ cảnh và bỏ qua bootstrap nặng khi fingerprint runtime không đổi.

Deployment cũ được chép lên máy mà không có `.git` cần nâng cấp một lần bằng
`--migrate-existing`. Trình cài clone source mới vào staging trước, dừng stack
sạch, giữ nguyên `.env.native`, `var` và `.venv-native`, rồi lưu source cũ ở một
đường dẫn backup được in ra; không tự xóa backup:

```bash
curl -fsSL https://raw.githubusercontent.com/ngucungcode/thuyet-minh-offline-gpu/v0.1.2/install.sh \
  | sudo bash -s -- --ref v0.1.2 --migrate-existing \
      --profile auto --start --yes
```

Chạy riêng `sudo bash -s -- ...` không tải installer và vì vậy không thực hiện
gì; luôn giữ phần `curl ... |` ở đầu lệnh.

Trong lúc đổi source, installer ghi journal atomic cạnh thư mục cài và rollback
khi nhận `INT`/`TERM` hoặc khi bước sau lỗi. Nếu máy mất điện đúng lúc rename,
lần chạy sau sẽ dừng fail-closed khi thấy journal, thay vì coi source thiếu dữ
liệu là một bản cài hợp lệ.

## Trạng thái

Phase 1 cung cấp nền Docker/GPU và acquisition. Phase 2 nối transcript ưu tiên
phụ đề và fallback ASR CUDA bằng `faster-whisper`. Phase 3 dịch offline sang
tiếng Việt bằng Gemma 4 qua `llama.cpp` CUDA. Phase 4 tách thoại diễn viên bằng
TIGER-DnR, giữ nhạc/hiệu ứng, tổng hợp giọng Việt bằng VieNeu hoặc Piper, fit
timeline 48 kHz rồi mux một track AAC với video passthrough. Mỗi stage có
checkpoint atomic để tiếp tục sau lỗi/process restart. Phase 5 bổ sung nghiệm
thu native, CycloneDX SBOM và cleanup artifact có retention/dry-run an toàn.

Các số đo Phase 4 (thời gian, VRAM, loudness và sai lệch A/V) chỉ được công bố
sau khi `native-phase4-acceptance.sh` chạy xong trên GPU đích; README không suy
diễn benchmark từ unit test hoặc fixture giả.

## Chạy trực tiếp trên GPU container hiện tại

Đây là chế độ triển khai chính cho máy thuê không có Docker daemon/systemd. Source,
virtualenv và dữ liệu bền vững nằm dưới `/workspace/thuyet-minh-offline`; PyTorch/CUDA
do nhà cung cấp cài sẵn được giữ nguyên, còn CTranslate2/faster-whisper được cài vào
`.venv-native`. Supervisor quản lý đúng một tiến trình cho API, worker, Prowlarr và
qBittorrent.

Trạng thái đã nghiệm thu đến ngày 02/08/2026 trên máy hiện tại:

- RTX 3090 24 GiB, driver 580.82.07, CUDA 12.6 và PyTorch 2.7.0: PASS.
- CTranslate2 4.8.1 có `float16`/`int8_float16`: PASS.
- 314 unit/integration test trên Ubuntu, GPU kernel smoke và qBittorrent 4.x
  pause/resume smoke: PASS. Windows đạt 312 test và skip đúng 2 test phụ thuộc
  symlink/GPU của Linux.
- Hai model ASR đã cài và xác minh: Small (~484 MB) và Large-v3-Turbo
  (~1,62 GB). Cả hai đạt transcript 7/7 token trên fixture tự sinh.
- Hai model dịch đã cài và xác minh SHA-256: Gemma 4 E2B Q4 (~3,35 GB) và
  Gemma 4 31B Q4 (~17,65 GB). Bản 31B dùng khoảng 18,67 GiB VRAM ở context
  2.048 và là model mặc định.
- `llama.cpp` `b10208` được build CUDA sm_86 từ commit đầy đủ đã khóa, tắt
  CURL và Web UI; server chỉ bind cổng loopback trong lúc dịch.
- API `8080`, qBittorrent WebUI `8081` và Prowlarr `9696` chỉ bind
  `127.0.0.1`: PASS.
- API/worker tự khởi động lại sau SIGTERM; SQLite dùng WAL: PASS.

Các lệnh vận hành trên máy thuê:

```bash
cd /workspace/thuyet-minh-offline
./scripts/native-stack.sh status
./scripts/native-stack.sh logs 200
.venv-native/bin/dub health
./scripts/native-acceptance.sh
./scripts/native-phase2-acceptance.sh asr-faster-whisper-small
./scripts/native-phase3-acceptance.sh mt-gemma4-31b-q4
./scripts/native-phase4-acceptance.sh
```

Cài mới vào một GPU container Ubuntu 22.04 tương đương:

```bash
cd /workspace/thuyet-minh-offline
./scripts/native-bootstrap.sh
./scripts/native-stack.sh start
./scripts/native-init-services.sh --rotate-secrets
./scripts/native-acceptance.sh
```

`native-bootstrap.sh` pin qBittorrent từ Ubuntu Jammy và kiểm SHA-256 gói Prowlarr
trước khi giải nén. `native-init-services.sh` khóa hai WebUI về loopback, tạo secret
ngẫu nhiên và lưu chúng dưới `var/secrets` với mode `0600`; script không in giá trị
secret. Tài khoản qBittorrent là `dub`; mật khẩu hiện tại nằm trong
`var/secrets/qbittorrent_password` và chỉ nên đọc trong phiên SSH quản trị.

Truy cập ba giao diện từ máy cá nhân bằng tunnel, không mở cổng quản trị ra Internet:

```bash
ssh -p <SSH_PORT> \
  -L 8080:127.0.0.1:8080 \
  -L 8081:127.0.0.1:8081 \
  -L 9696:127.0.0.1:9696 \
  root@<GPU_HOST>
```

Sau đó dùng dashboard ở `http://127.0.0.1:8080/`, API docs ở
`http://127.0.0.1:8080/docs`, qBittorrent ở `http://127.0.0.1:8081` và Prowlarr
ở `http://127.0.0.1:9696`. Dashboard bao phủ tìm/chọn nguồn, cấu hình model,
chọn phụ đề/ngôn ngữ khi pipeline yêu cầu, theo dõi job, hủy/tiếp tục và tải
MP4/SRT/timing report. Prowlarr hiện chưa
có indexer; quản trị viên phải tự thêm một indexer hợp pháp/content-agnostic trước
khi tìm được nguồn thật. Không có tracker/indexer nào được đóng gói hoặc gợi ý sẵn.

Hướng dẫn chi tiết để thêm indexer trong giao diện Prowlarr và đăng nhập
OpenSubtitles API mà không lưu mật khẩu nằm tại
[`docs/WEB_WORKFLOW.md`](docs/WEB_WORKFLOW.md). Các thao tác tích hợp chỉ chạy
qua loopback/SSH tunnel; không công khai cổng quản trị ra Internet.

Supervisor sống qua việc ngắt SSH nhưng không tự sống lại khi nhà cung cấp khởi động
lại toàn bộ container. Khi đó chạy `scripts/native-stack.sh start`, hoặc đặt startup
command của nhà cung cấp thành:

```bash
/workspace/thuyet-minh-offline/scripts/native-stack.sh foreground
```

Container được quản lý này không có `CAP_NET_ADMIN`, vì vậy worker native không thể
có network namespace tương đương `network_mode: none`. Worker chỉ mở model theo
đường dẫn local đã kiểm SHA, đặt toàn bộ biến Hugging Face offline và cài audit hook
chặn DNS/TCP IPv4/IPv6, ngoại trừ đúng cổng loopback cố định của `llama-server`.
Đây là guard ở lớp tiến trình Python, không phải cô lập egress cấp kernel; cấu hình
Docker vẫn dùng `network_mode: none` để đạt lớp kernel.

## Chạy bằng Docker Compose (tùy chọn)

Chế độ Docker cũ vẫn được giữ cho một Ubuntu host đầy đủ có Docker daemon và NVIDIA
Container Toolkit. Nó không được dùng bên trong GPU container thuê hiện tại.

### Yêu cầu máy chủ

- Ubuntu 24.04 x86_64, Docker Engine và Docker Compose.
- NVIDIA Container Toolkit đã cấu hình cho Docker.
- NVIDIA driver `>=570.26`, compute capability `>=7.0`; model mặc định 31B cần
  GPU 24 GiB, còn E2B là lựa chọn cho GPU nhỏ hơn.
- Ít nhất 16 GiB RAM và dung lượng trống phù hợp với media/model.

Image ứng dụng khóa CUDA 12.8/cuDNN 9 theo digest. Worker kiểm `nvidia-smi`,
thực thi một CUDA matrix multiplication bằng PyTorch và kiểm compute type CUDA
của CTranslate2 trước khi nhận việc.

## Cấu hình ban đầu

```bash
cd gpu-server
cp .env.example .env
cp secrets/prowlarr_api_key.example secrets/prowlarr_api_key.txt
cp secrets/qbittorrent_password.example secrets/qbittorrent_password.txt
cp secrets/opensubtitles_api_key.example secrets/opensubtitles_api_key.txt
cp secrets/opensubtitles_token.example secrets/opensubtitles_token.txt
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
sed -i "s/^PUID=.*/PUID=${HOST_UID}/; s/^PGID=.*/PGID=${HOST_GID}/" .env
chown "${HOST_UID}:${HOST_GID}" secrets/*.txt
chmod 600 secrets/*.txt
```

Giữ cùng `PUID`/`PGID` cho API, worker, model-manager và qBittorrent. Hai giá
trị phải trùng chủ sở hữu các file secret trên host vì Compose mount secret từ
file và tiến trình API không chạy root. Mặc định `10001` chỉ phù hợp khi host
thực sự sở hữu file bằng UID/GID đó. Sau khi đổi UID/GID phải rebuild image.

Build image nền rồi khởi động riêng hai giao diện quản trị:

```bash
docker compose build api
docker compose up -d prowlarr qbittorrent
```

- Mở Prowlarr tại `http://127.0.0.1:9696`, tự thêm indexer mà bạn được phép sử dụng.
- Mở qBittorrent tại `http://127.0.0.1:8081`, đổi mật khẩu WebUI cho trùng secret
  và đặt `Default Save Path` thành `/data/incoming`.
- Cổng peer TCP/UDP `${TORRENTING_PORT:-6881}` được publish để torrent có thể
  nhận kết nối vào; WebUI vẫn chỉ bind vào loopback.
- Tạo ứng dụng Prowlarr trỏ tới qBittorrent bằng địa chỉ nội bộ
  `http://qbittorrent:8080`, không dùng cổng loopback `8081`; không bật tự động
  tải ngoài ý muốn.
- Cổng peer `6881` bind mọi interface. Chỉ mở nó qua firewall khi cần; không
  publish API/WebUI ra LAN nếu chưa bổ sung xác thực và TLS.
- OpenSubtitles là tùy chọn. Nếu dùng, điền API key và bearer token của tài khoản
  vào hai file secret tương ứng, đồng thời đặt `DUB_OPENSUBTITLES_URL` trong
  `.env` đúng bằng `base_url` mà endpoint login trả về (host thường hoặc VIP),
  rồi force-recreate
  API. Token có thể hết hạn và phải được thay thủ công. Nếu thiếu key/token hoặc
  URL không thuộc allowlist chính thức, hệ thống chỉ dùng subtitle nhúng/sidecar
  rồi fallback ASR.

Sau khi lấy API key Prowlarr và đã đổi mật khẩu qBittorrent, thay `REPLACE_ME`
trong các file secret, chạy lại lệnh `chown`/`chmod` ở trên rồi khởi động ứng
dụng. Mỗi lần đổi secret phải recreate API để mount lại giá trị:

```bash
docker compose config --quiet
docker compose run --rm --no-deps worker python3.12 -m dub_server.worker --once
docker compose up -d --force-recreate api worker
curl -fsS http://127.0.0.1:8080/v1/health | \
  jq -e '.status == "ok" and .gpu.ready == true and .acquisition_configured == true'
```

OpenAPI ở `http://127.0.0.1:8080/docs`. Worker dùng `network_mode: none`, vì
vậy chỉ đọc job và artifact qua SQLite/volume dùng chung sau khi tải hoàn tất.
Service một-lần `volume-init` tạo các thư mục dùng chung với UID/GID đã cấu hình
trước khi API, worker hoặc qBittorrent khởi động. API, Prowlarr, qBittorrent và
model-manager dùng các mạng egress riêng; qBittorrent chỉ được mount volume
`incoming`, không được ghi vào `jobs` hoặc `output`; worker chỉ đọc `incoming`.
API phải chạy đúng một process Uvicorn như cấu hình Compose hiện tại; không tăng
`--workers` nếu chưa thay khóa mutation nội bộ bằng lease liên tiến trình.

## CLI/API đầy đủ

```bash
dub version
dub doctor
dub stack start
dub stack status
dub stack logs --lines 200
dub models list
dub models profiles
dub models recommend
dub models install-profile maximum --yes
dub search "Tên phim" --year 2024
dub submit --release-id RELEASE_ID --i-have-rights \
  --asr-model asr-faster-whisper-small \
  --translation-model mt-gemma4-31b-q4 \
  --separation-model separation-tiger-dnr \
  --tts-model tts-vieneu-v2 --wait -o result.mp4
dub jobs list --limit 20
dub watch JOB_ID --fetch-dir ./results
dub events JOB_ID
dub status JOB_ID
dub subtitle-select JOB_ID SUBTITLE_ID
dub subtitle-use-asr JOB_ID
dub language-select JOB_ID en
dub cancel JOB_ID
dub resume JOB_ID
dub fetch JOB_ID --kind all -o ./results
dub maintenance cleanup
dub maintenance cleanup --apply --yes
dub maintenance sbom
```

`dub watch` dùng SSE, tự reconnect bằng event cursor và hiển thị stage, phần trăm,
tốc độ tải, ETA và số block. `dub fetch` ghi qua `.part`, hỗ trợ HTTP Range và chỉ
publish file đích atomically. Mọi lệnh destructive đều yêu cầu cờ xác nhận rõ.

API chính nằm dưới `/v1`: `health`, `capabilities`, `models`, `search`, `jobs`,
`jobs/{id}`, chọn subtitle/ASR, chọn lại ngôn ngữ, `cancel`, `resume` và stream tiến độ SSE. Tạo job bắt buộc gửi
`rights_confirmed: true`; `release_id` phải xuất phát từ một lần tìm kiếm hợp lệ,
không nhận URL tải tùy ý từ client. ID model được kiểm tra đúng stage theo catalog.
Trạng thái `cancelling` của job đang chạy tiếp tục giữ slot cho tới khi
qBittorrent xác nhận pause. Job vốn đã `paused`/`failed` không giữ slot trong lúc
hủy, và hệ thống không gửi lệnh pause muộn nếu một job khác đang sở hữu backend.
Không công bố hủy hoàn tất khi trạng thái backend còn chưa chắc chắn. `resume`
chỉ nhận job `paused` hoặc lỗi có cờ retry; stage có artifact/checkpoint hợp lệ
không chạy inference lại.

Theo dõi chi tiết bằng `dub status JOB_ID` hoặc SSE (trường `progress_permille`
chạy từ 0 đến 1000 và event ghi rõ stage):

```bash
curl -N -H 'Accept: text/event-stream' \
  http://127.0.0.1:8080/v1/jobs/JOB_ID/events
```

Luồng transcript cố định của Phase 2:

```text
READY_OFFLINE
  ├─ subtitle → parse SRT/VTT/ASS → source-transcript.json
  └─ asr → FFmpeg PCM mono 16 kHz → faster-whisper CUDA → source-transcript.json
       └─ confidence < 0,5 → NEEDS_LANGUAGE → language-select → chạy lại local

source-transcript.json + SQLite segments/checkpoint → READY_TRANSLATION
  → Gemma 4/llama.cpp → translated-transcript.json → READY_TTS
  → TIGER-DnR: dialogue + music + effects
      ├─ bỏ dialogue gốc
      └─ music + effects → accompaniment.wav
  → VieNeu/Piper → fit từng slot → narration-48k.wav
  → loudness + ducking → video copy + AAC → JOB_ID.mp4
  → xác minh đúng 1 video track + 1 audio track → COMPLETED
```

Timestamp dùng số nguyên microsecond, được clamp theo duration và ép không overlap.
Artifact JSON ghi qua file tạm + `os.replace`, lưu SHA-256 trong checkpoint và được
dùng lại sau restart mà không chạy inference lần nữa. Khi hủy offline, worker
giải phóng subprocess/model và xóa file `.part` trước khi chốt `CANCELLED`; nguồn
torrent trong `incoming` không bị cleanup artifact đụng tới.

### Nghiệm thu native Phase 4

Cài và verify đủ model trước khi chạy. Lệnh không truyền đối số dùng fixture ngắn
tự sinh; lệnh có `--input` dùng một clip local mà người vận hành có quyền xử lý:

```bash
./scripts/native-model.sh install separation-tiger-dnr
./scripts/native-model.sh install tts-vieneu-v2
./scripts/native-model.sh install tts-neucodec-onnx-int8
./scripts/native-model.sh verify separation-tiger-dnr
./scripts/native-model.sh verify tts-vieneu-v2
./scripts/native-model.sh verify tts-neucodec-onnx-int8
./scripts/native-phase4-acceptance.sh
./scripts/native-phase4-acceptance.sh --quick --quick-duration-seconds 10
./scripts/native-phase4-acceptance.sh --quick --quick-duration-seconds 300
./scripts/native-phase4-acceptance.sh --quick --quick-duration-seconds 1800
./scripts/native-phase4-acceptance.sh \
  --input /data/incoming/authorized-test.mkv --clip-duration-seconds 12
```

Script tạm dừng worker, ép toàn bộ runtime model sang offline, kiểm separation,
TTS, sample count/timing, loudness, MP4 track contract và ghi báo cáo vào
`var/state/phase4-acceptance.json`; worker được khởi động lại bằng trap kể cả khi
nghiệm thu lỗi. Không coi fixture nhanh là benchmark phim dài.

Kết quả nghiệm thu RTX 3090 mới nhất nằm trong `PHASE4_REPORT.md` và
`PHASE5_REPORT.md`; ma trận 10 giây, 5 phút và 30 phút đều đã chạy thật.

### SBOM và cleanup Phase 5

SBOM dùng đúng Python distributions đang cài cộng với lock manifest của model,
native và toàn bộ dependency web/npm, sau đó ghi CycloneDX 1.6 atomically. CI
xác minh file phát hành bằng schema CycloneDX 1.6 chính thức:

```bash
.venv-native/bin/python scripts/generate-sbom.py \
  --models-lock config/models.lock.json \
  --native-lock native/components.lock.json \
  --web-lock web/package-lock.json \
  --output var/reports/sbom.cdx.json
```

Cleanup mặc định chỉ dry-run. Nó chỉ lập kế hoạch cho thư mục/tệp mang đúng ID
job dưới `jobs` và `output`: job đã hủy được dọn ngay, job lỗi retry giữ 7 ngày.
Job đang chạy, job hoàn tất, lỗi không retry và toàn bộ `incoming` đều bị loại
khỏi phạm vi. Xem JSON dry-run trước, rồi mới thêm `--apply` nếu đúng mục tiêu:

```bash
.venv-native/bin/python scripts/cleanup-job-artifacts.py
.venv-native/bin/python scripts/cleanup-job-artifacts.py --apply
```

Khi dùng `--apply`, cleanup khóa transaction SQLite, đọc lại revision/status và
retention ngay trước khi xóa toàn bộ action của từng job. Nếu job vừa được
resume sau lúc lập plan, cleanup bỏ qua job đó.

## Kiểm thử cục bộ

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

Các test acquisition dùng HTTP mock và không liên hệ indexer, torrent hoặc
OpenSubtitles thật. Máy không đạt cấu hình GPU vẫn chạy được unit test, nhưng
không được coi là đã vượt cổng GPU production.

## Model

`config/models.lock.json` là catalog bất biến. ASR, Gemma 4, TIGER-DnR, VieNeu v2,
NeuCodec ONNX int8 và Piper tiếng Việt đều pin revision, size, SHA-256 từng file
và tree hash. VieNeu v2 là lựa chọn TTS chất lượng cao mặc định và cần cài thêm
entry hỗ trợ `tts-neucodec-onnx-int8`; Piper là fallback CPU có thể chọn độc lập.
TIGER-DnR là model separation mặc định. Trạng thái cài thật luôn lấy từ receipt
verify trên máy đang chạy, không suy ra từ catalog.

Model chỉ được tải qua process model-manager có egress; worker không import trình
tải và luôn băm lại toàn bộ tree trước inference:

```bash
cd /workspace/thuyet-minh-offline
./scripts/native-model.sh list
./scripts/native-model.sh install mt-gemma4-31b-q4
./scripts/native-model.sh verify mt-gemma4-31b-q4
./scripts/native-model.sh install separation-tiger-dnr
./scripts/native-model.sh install tts-piper-vi-vais1000-medium
./scripts/native-model.sh verify tts-piper-vi-vais1000-medium
```

Receipt ngoài model tree giúp API hiển thị lần xác minh gần nhất mà không băm lại
hàng GB mỗi request; worker không tin receipt và vẫn xác minh file/size/hash thật.

## Giấy phép

Mã nguồn được phát hành theo
[GPL-3.0-or-later](LICENSE). Attribution, giấy phép runtime/model và nghĩa vụ phân
phối tương ứng nằm trong [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Người
dùng chỉ được tải và xử lý nội dung/giọng nói mà mình sở hữu, được cấp phép hoặc
có quyền sử dụng hợp pháp. Bản CycloneDX sinh từ runtime GPU đã nghiệm thu nằm tại
[release/sbom.cdx.json](release/sbom.cdx.json).
