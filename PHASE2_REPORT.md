# Báo cáo Phase 2 — transcript ưu tiên phụ đề và ASR CUDA

Ngày nghiệm thu native: 31/07/2026<br>
Đường dẫn triển khai: `/workspace/thuyet-minh-offline`

## Kết luận

Phase 2 đạt cổng trên RTX 3090 thật. Worker đã chuyển từ heartbeat sang xử lý job
`READY_OFFLINE`: phụ đề SRT/VTT/ASS hợp lệ đi thẳng thành transcript; nếu không có
phụ đề phù hợp, worker decode PCM mono 16 kHz và chạy `faster-whisper` bằng
CTranslate2 CUDA. Kết quả được ghi JSON atomic, lưu segment/timestamp microsecond
trong SQLite và chuyển sang `READY_TRANSLATION` cho Phase 3.

Hai model ASR đã cài, xác minh và smoke thật: `asr-faster-whisper-small` và
`asr-faster-whisper-large-v3-turbo`. `asr-faster-whisper-large-v3` đã pin đầy đủ
nhưng chưa tải. Dịch, TTS và mux chưa được triển khai; output MP4 thuyết minh vẫn
chưa phải Definition of Done ở giai đoạn này.

## Quyết định và đánh đổi

- Chọn `faster-whisper 1.2.1` + `CTranslate2 4.8.1` CUDA thay vì thêm một runtime
  whisper.cpp thứ hai. Runtime này đã được Phase 1 xác minh trên máy thuê và hỗ trợ
  `float16`/`int8_float16`.
- Mặc định dùng Large-v3-Turbo `float16` để cân bằng chất lượng/tốc độ; Small là
  lựa chọn nhẹ. Large-v3 đầy đủ dành cho người dùng muốn tải thêm model lớn.
- Subtitle hợp lệ tuyệt đối không resolve model, không decode audio và không dựng
  recognizer. Subtitle hỏng trong mode `prefer` mới fallback ASR.
- Worker chỉ nhận đường dẫn model local tuyệt đối sau khi kiểm size/SHA-256/tree;
  `local_files_only=True`. Chỉ model-manager có quyền gọi `snapshot_download`.
- Native managed container không có `CAP_NET_ADMIN`. Audit hook chặn DNS/TCP ở lớp
  Python và biến Hugging Face offline đã đạt canary; đây không thay thế network
  namespace. Docker Compose vẫn dùng `network_mode: none` cho worker.
- Phase 2 ghi checkpoint artifact hoàn chỉnh; resume sau khi JSON đã publish không
  inference lại. Checkpoint theo chunk giữa một phim dài được để cho Phase 5, nên
  process chết giữa lần ASR đầu tiên vẫn phải chạy lại stage ASR.

## Model lock và cài đặt

| Model | Revision | Tree SHA-256 | Trạng thái |
|---|---|---|---|
| faster-whisper-small | `536b0662742c02347bc0e980a01041f333bce120` | `101416fafddb7e7d39f44ef221e4ad313836b7a1d740fe2509a9d9ab0042959f` | installed, valid |
| faster-whisper-large-v3-turbo | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` | `be36aea51ef822e221e44f708874ee6951df19b82d669577c167104632a499ab` | installed, valid |
| faster-whisper-large-v3 | `edaa852ec7e145841d8ffdb056a99866b5f0a478` | `f7aae5248606925aecbc60bb504bf275475e631893cf156ffb8acd1dcf478736` | pinned, not installed |

API `/v1/models` hiện báo Small và Turbo `installed=true`, `valid=true`, kèm
`verified_at`. Receipt này chỉ phục vụ control plane; worker băm lại model trước
mỗi lần mở.

## Kết quả kiểm thử

### Suite

- Windows/local: **164 passed, 1 skipped**. Test bị skip là integration FFmpeg vì
  host Windows không có FFmpeg trong PATH; phần tương ứng chạy đạt trên Ubuntu.
- Ubuntu native sau deploy: toàn bộ suite PASS.
- Python compile: PASS.
- Supervisor sau deploy: API, worker, Prowlarr và qBittorrent đều `RUNNING`.
- API health: `status=ok`, `gpu.ready=true`, SQLite `wal`.

### Fixture tự sinh hợp pháp

Bài nghiệm thu dùng FFmpeg `testsrc2` + `flite`, không tải media bên ngoài. Câu nói:

`this offline fixture tests local speech recognition`

| Chỉ số | Small | Large-v3-Turbo |
|---|---:|---:|
| Token khớp | 7/7 | 7/7 |
| Language | `en` | `en` |
| Language probability | 0,9961 | 0,9990 |
| ASR stage | 2,503 giây | 3,219 giây |
| RTF trên clip 8 giây | 0,313× | 0,402× |
| Peak GPU process memory | 882–914 MiB | 2.002 MiB |
| Peak RSS | khoảng 1.040 MiB | khoảng 2.122 MiB |
| Timestamp | bounded, monotonic | bounded, monotonic |

Turbo và Small đều đạt `READY_TRANSLATION`, tạo một segment và transcript đúng câu.
Route subtitle tạo một segment với `asr_invocations=0`. Worker subprocess thật cũng
đạt `READY_TRANSLATION`, ghi artifact/checkpoint và dừng SIGTERM với exit code 0.

### Offline guard

- `socket.getaddrinfo`: blocked.
- IPv4 `socket.connect`: blocked.
- IPv6 `socket.connect`: blocked.
- Model được load từ directory đã verify, không dùng repo ID và đặt
  `local_files_only=True`.
- Managed native: PASS ở lớp Python, không tuyên bố cô lập kernel.
- Docker worker: cấu hình `network_mode: none` vẫn còn nguyên.

Báo cáo máy thật được lưu tại:

- `var/state/phase2-acceptance.json` (Turbo)
- `var/state/phase2-acceptance-small.json` (Small, gồm worker subprocess smoke)

## API/trạng thái mới

- `POST /v1/jobs/{id}/language` và CLI `dub language-select JOB_ID en`.
- `READY_OFFLINE → TRANSCRIBING|SUBTITLE_SELECTED → READY_TRANSLATION`.
- Confidence ngôn ngữ tự động `<0,5` chuyển `NEEDS_LANGUAGE`.
- Cancel offline không pause torrent đã hoàn tất. Worker chỉ chốt `CANCELLED` sau
  khi decoder/recognizer trả về và model đã được giải phóng.
- Resume lỗi ASR không gọi qBittorrent dù job còn giữ `task_id` lịch sử.

## Lệnh tái kiểm thử

```bash
cd /workspace/thuyet-minh-offline

# Unit/integration suite
.venv-native/bin/python -m compileall -q src
.venv-native/bin/python -m pytest -q

# Verify model đã cài
source scripts/native-common.sh
runuser -u dub --preserve-environment -- \
  .venv-native/bin/python -m dub_server.model_manager verify \
  asr-faster-whisper-small \
  --lock "$DUB_MODELS_LOCK_PATH" --models-dir "$DUB_MODELS_DIR"

# Nghiệm thu GPU/offline/worker; script luôn bật lại worker bằng trap
./scripts/native-phase2-acceptance.sh asr-faster-whisper-small
./scripts/native-phase2-acceptance.sh asr-faster-whisper-large-v3-turbo

# Health và catalog
curl -fsS http://127.0.0.1:8080/v1/health | jq
curl -fsS http://127.0.0.1:8080/v1/models | jq '.models[0:3]'
```

## Việc chưa thuộc Phase 2

1. Chưa dịch transcript sang tiếng Việt; `READY_TRANSLATION` là điểm dừng có chủ ý.
2. Chưa TTS, fit timing, thay audio hoặc xuất MP4/SRT tiếng Việt.
3. Chưa benchmark corpus phim dài/đa ngôn ngữ; fixture chỉ là smoke deterministic.
4. Chưa checkpoint ASR theo chunk giữa video dài hoặc tự resume sau SIGKILL.
5. Prowlarr chưa có indexer hợp pháp và OpenSubtitles chưa có credential, nên chưa
   chạy acquisition nội dung thật. Người dùng phải tự cung cấp nguồn được phép.

## Cổng Phase 3

Phase 2 đã đạt và phải dừng tại `READY_TRANSLATION`. Phase 3 chỉ bắt đầu sau khi
người dùng xác nhận tiếp tục; khi đó cần pin/cài ít nhất hai model dịch local, mapping
ngôn ngữ và benchmark dịch sang tiếng Việt trên RTX 3090.
