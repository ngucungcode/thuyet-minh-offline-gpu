# Phase 4 — Tách thoại, TTS, khớp thời lượng và xuất MP4

Ngày nghiệm thu: 01/08/2026<br>
Máy nghiệm thu: NVIDIA GeForce RTX 3090 24 GiB, driver 610.43.02, CUDA 12.6

## Kết luận

**PASS cổng kỹ thuật native GPU.** Pipeline đã chạy hoàn toàn cục bộ theo luồng:

`MP4 → TIGER-DnR → bỏ dialogue → giữ music+effects → VieNeu/Piper → timeline 48 kHz → ducking/mix → H.264 copy + AAC`

Output được xác minh có đúng một video H.264 và một audio AAC 48 kHz stereo,
không map audio nguồn. Tất cả mốc 10 giây, 5 phút và 30 phút có duration error,
A/V duration error và start-time sync error bằng 0 µs trên fixture nghiệm thu.

## Model đã khóa và xác minh

| Vai trò | Model | Revision | Tree SHA-256 |
|---|---|---|---|
| Separation | `separation-tiger-dnr` | `b7a59560bbca10febbcd46fb01600f868e587f57` | `c6995ec4b397c21d149c55cd8cd4e5fc0d567e5846d2309a5b78a6fa48d54333` |
| TTS mặc định | `tts-vieneu-v2` | `b62b1cbddec67cb1d26ac602965d39f0a7faddf2` | `fa8be1d47bc19d06231736ad673dddb38e04855f7c7160e71f74cc5064b9eb9c` |
| VieNeu codec | `tts-neucodec-onnx-int8` | `706f4bd5fcc39b039c333d5407f58b0075dcee07` | `be786eaca4e9d070d99e845f6b62fdec7ab91c1531225659a72a2764ae9e6911` |
| TTS nhẹ | `tts-piper-vi-vais1000-medium` | `320d5f7f7751a17ef6512d5c23863056c6a11c0f` | `a4ea48bcaf4c9ff6da8abdeee79da812be9a785ef6c9210ec9bcda51f4855333` |

Model manager đã băm lại từng tree trên máy production trước inference; cả bốn
model trả `valid=true`. Worker không được tải model hoặc mở kết nối Internet.

## Kết quả đo thực tế với VieNeu mặc định

| Thời lượng | Tổng thời gian | RTF | Separation | TTS | Mix/export | GPU peak | RSS child peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 giây | 35,547 s | 3,555× | 10,131 s | 23,160 s | 0,951 s | 2.544 MiB | 2.322,9 MiB |
| 5 phút | 167,095 s | 0,557× | 114,085 s | 22,047 s | 24,138 s | 3.222 MiB | 2.327,1 MiB |
| 30 phút | 886,841 s | 0,493× | 666,198 s | 42,079 s | 143,442 s | 3.266 MiB | 2.328,8 MiB |

Cổng RTF ≤4× và RSS ≤2,5 GiB đều đạt. TIGER đọc theo chunk 120 giây với
context 4 giây nên RAM/VRAM không tăng theo toàn bộ thời lượng phim.

| Mốc | Integrated loudness | True peak | MP4 SHA-256 |
|---|---:|---:|---|
| 10 giây | −18,1 LUFS | −2,6 dBFS | `7f88410ba2b9570b480fbe1fcf815776d71314d19f6e9e00624245a2914b2e74` |
| 5 phút | −22,3 LUFS | −3,1 dBFS | `affe9dd33f8fbf967fa1c16bd500d3e6a3314cf3c50708073ea9393ee2ad8544` |
| 30 phút | −22,4 LUFS | −4,8 dBFS | `e911f75921cc11639a282ba289868f1c39b689466d17f154ab3f32de55f55795` |

Loudness dài thấp hơn target narration vì fixture chỉ có một câu thuyết minh và
phần lớn timeline là nhạc/SFX nền. Không có clipping.

## TTS có thể chọn

- VieNeu persistent giữ model/codec resident, nạp đúng một lần cho cả hai lượt
  đo tốc độ và toàn bộ block. Test 4 giây: 41,185 s, GPU peak 2.514 MiB.
- Piper được tìm trực tiếp cạnh interpreter virtualenv và chạy được khi PATH
  Supervisor tối giản. Test 4 giây: 14,428 s, RTF 3,607×, sync 0 µs.
- Lỗi trước nghiệm thu do resolve `.venv-native/bin/python` thành
  `/usr/bin/python` đã được sửa; regression test bảo đảm giữ symlink virtualenv.

## Độ bền và an toàn artifact

- Cancel/timeout gửi tín hiệu cho toàn bộ process group TIGER/VieNeu, gồm FFmpeg
  con; không để tiến trình mồ côi giữ file hoặc GPU.
- Checkpoint export lưu SHA-256 và identity của translation, separation,
  timeline, TTS model. Restart sau checkpoint dùng lại MP4 đã niêm phong; file
  bị sửa hoặc symlink sẽ bị từ chối và export lại.
- Endpoint tải artifact chỉ phục vụ đường dẫn xác định dưới configured root,
  regular file không symlink và SHA-256/size trùng result đã niêm phong.

## Phạm vi đánh giá chất lượng

Fixture tổng hợp kiểm tra hợp đồng media, thời lượng, tài nguyên và việc dialogue
stem không được publish. Nó không thay thế kiểm nghe corpus phim có bản quyền để
chấm độ rò lời diễn viên hoặc độ tự nhiên của giọng. Việc chấm nghe đó vẫn cần
thực hiện trên nội dung mà người vận hành có quyền sử dụng.

## Artifact báo cáo

- `var/state/phase4-acceptance-vieneu.json`
- `var/state/phase4-acceptance-piper.json`
- `var/state/phase4-acceptance-10s-vieneu.json`
- `var/state/phase4-acceptance-300s-vieneu.json`
- `var/state/phase4-acceptance-1800s-vieneu.json`
