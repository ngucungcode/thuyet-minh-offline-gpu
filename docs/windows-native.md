# Windows 10 native

Tài liệu này cài Thuyết Minh Offline GPU trực tiếp trên Windows 10, không dùng
WSL2 hoặc Docker. Bản Windows hiện là MVP local-upload: dashboard, API, worker và
toàn bộ pipeline GPU chạy native; người dùng tải MP4/MKV và SRT từ trình duyệt.
Prowlarr và qBittorrent không được installer Windows cài hay quản lý.

## Yêu cầu

- Windows 10 22H2 x64, build 19045 trở lên.
- NVIDIA RTX 20, 30, 40 hoặc 50 có ít nhất 6 GiB VRAM.
- NVIDIA driver tối thiểu 560.76 với CUDA 12.6 hoặc 570.65 với CUDA 12.8.
- CUDA Toolkit 12.6 hoặc 12.8; RTX 50 (`sm_120`) bắt buộc 12.8.
- cuDNN 9 cho CUDA 12.x để chạy lớp convolution của Whisper/CTranslate2. Installer
  ưu tiên DLL trong PyTorch `cu128` và sẽ từ chối hoàn tất nếu probe CUDA không đạt.
- Python 3.11 hoặc 3.12 x64, có `python.exe` trong `PATH`.
- Git for Windows, CMake, Ninja và FFmpeg (`ffmpeg.exe`, `ffprobe.exe`) trong `PATH`.
- Visual Studio 2022 Build Tools với workload **Desktop development with C++**.
- RAM tối thiểu 16 GiB và dung lượng trống tối thiểu 25/35/55 GiB tương ứng profile
  `minimal`/`balanced`/`maximum`, chưa tính phim và output.

Nguồn cài chính thức: [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive),
[Python for Windows](https://www.python.org/downloads/windows/),
[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/),
[Git for Windows](https://git-scm.com/download/win),
[CMake](https://cmake.org/download/) và [FFmpeg](https://ffmpeg.org/download.html).
Ninja có thể cài bằng `winget install Ninja-build.Ninja` nếu máy có WinGet.

| Dòng GPU | CUDA target | Trạng thái Windows | Profile |
|---|---|---|---|
| RTX 20 | `sm_75` | Hỗ trợ | Theo VRAM |
| RTX 30 | `sm_86` | Hỗ trợ | Theo VRAM |
| RTX 40 | `sm_89` | Hỗ trợ | Theo VRAM |
| RTX 50 | `sm_120` | Thử nghiệm, cần CUDA 12.8 | Chỉ `minimal` |

`auto` chọn `minimal` từ 6 GiB, `balanced` từ 8 GiB và `maximum` từ 22 GiB.
RTX 50 luôn bị khóa ở `minimal` cho đến khi có báo cáo nghiệm thu phần cứng đầy đủ.

## Cài đặt

Clone repository và mở PowerShell trong thư mục vừa clone:

```powershell
git clone https://github.com/ngucungcode/thuyet-minh-offline-gpu.git
cd thuyet-minh-offline-gpu
Set-ExecutionPolicy -Scope Process Bypass
.\windows\preflight.ps1
.\windows\install.ps1 -Profile auto
```

Installer thực hiện các bước fail-closed sau:

1. Kiểm tra Windows build, toolchain, GPU, driver, VRAM và CUDA target thật.
2. Build `llama.cpp` CUDA từ commit đã khóa riêng cho kiến trúc card hiện tại.
3. Lấy TIGER và VieNeu đúng commit, rồi xác minh các file overlay bằng SHA-256.
4. Tạo `.venv-windows`, cài PyTorch 2.8.0 `cu128` và dependency đã khóa.
5. Chạy kernel FP16, CTranslate2, worker preflight và cài model đã chọn.

Không sao chép `.env.windows`, `.venv-windows` hoặc binary `llama.cpp` đã build sang
máy khác. Installer pin UUID, compute capability và CUDA Toolkit của đúng GPU.

Các tùy chọn hữu ích:

```powershell
# Chỉ dựng runtime, chưa tải model lớn
.\windows\install.ps1 -Profile minimal -SkipModels

# Chạy toàn bộ pytest sau khi cài
.\windows\install.ps1 -Profile auto -FullTest

# Giới hạn số job compile song song
.\windows\install.ps1 -Profile auto -BuildJobs 2
```

Muốn đặt model, job và output ở ổ khác, bỏ dấu `#` và sửa `DUB_NATIVE_ROOT` trong
`.env.windows` trước khi chạy installer lần đầu.

## Chạy và dừng

```powershell
.\windows\stack.ps1 start
.\windows\stack.ps1 status
.\windows\stack.ps1 logs -Lines 200
.\windows\stack.ps1 restart
.\windows\stack.ps1 stop
```

Mở [http://127.0.0.1:8080/](http://127.0.0.1:8080/) rồi chọn upload file cục bộ.
Stack chỉ bind loopback và quản lý hai tiến trình `api` + `worker`. PID được đối chiếu
với đường dẫn Python và thời điểm khởi động trước khi dừng; `taskkill /T` dọn cả cây
tiến trình con như FFmpeg và `llama-server.exe`.

Có thể dùng CLI qua wrapper để khỏi activate virtual environment:

```powershell
.\windows\dub.ps1 doctor
.\windows\dub.ps1 models list
.\windows\dub.ps1 jobs list
```

## Khắc phục sự cố

- `preflight.ps1` báo thiếu lệnh: đóng và mở lại PowerShell sau khi cài công cụ, rồi
  kiểm tra `Get-Command python, git, cmake, ninja, ffmpeg, nvcc, nvidia-smi`.
- Không tìm thấy Visual Studio: mở Visual Studio Installer, Modify Build Tools 2022
  và bật **Desktop development with C++** cùng MSVC x64/x86 build tools.
- RTX 50 báo thiếu `sm_120`: gỡ CUDA cũ khỏi đầu `PATH`, cài CUDA 12.8 rồi xác minh
  `nvcc --list-gpu-arch | Select-String compute_120`.
- PyTorch báo `no kernel image`: chạy
  `.\.venv-windows\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`.
  Installer yêu cầu wheel `cu128` và từ chối tiếp tục nếu kiến trúc card không có trong
  danh sách kernel của wheel.
- CTranslate2 báo thiếu `cudnn64_9.dll`: cài cuDNN 9 cho CUDA 12.x, bảo đảm thư mục
  chứa DLL nằm trong `PATH`, rồi chạy lại installer. Runtime cũng thêm thư mục
  `torch\lib` của wheel PyTorch vào đường tìm DLL.
- Dashboard không mở: chạy `.\windows\stack.ps1 status`, sau đó xem
  `.\windows\stack.ps1 logs -Lines 200`.
- Cần chạy trực tiếp để quan sát lỗi: dừng stack rồi chạy
  `.\windows\stack.ps1 foreground`; nhấn `Ctrl+C` để kết thúc.

Model chỉ được tải trong bước quản trị cài đặt. Sau khi model đã cài và xác minh,
worker xử lý ASR, dịch, tách thoại, TTS và dựng video bằng tài nguyên local.
