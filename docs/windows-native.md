# Windows 10 native

Tài liệu này cài Thuyết Minh Offline GPU trực tiếp trên Windows 10, không dùng
WSL2 hoặc Docker. Bản Windows hiện là MVP local-upload: dashboard, API, worker và
toàn bộ pipeline GPU chạy native; người dùng tải MP4/MKV và SRT từ trình duyệt.
Prowlarr và qBittorrent không được installer Windows cài hay quản lý.

## Yêu cầu

- Windows 10 22H2 x64, build 19045 trở lên.
- NVIDIA RTX 20, 30, 40 hoặc 50 có ít nhất 6 GiB VRAM.
- RAM tối thiểu 16 GiB và dung lượng trống tối thiểu 25/35/55 GiB tương ứng profile
  `minimal`/`balanced`/`maximum`, chưa tính phim và output.

Không cần cài prerequisite phần mềm trước. Bootstrap tự nâng quyền qua một hộp thoại
UAC, cài hoặc repair WinGet theo quy trình Microsoft, rồi cài phần còn thiếu: Python
3.12 x64, Git, CMake, Ninja, FFmpeg, Visual Studio 2022 C++ Build Tools, NVIDIA driver
và CUDA Toolkit 12.8. CUDA 12.6/12.8 tương thích đang có được giữ lại. cuDNN 9 được
lấy từ wheel PyTorch `cu128` và được kiểm tra bằng probe GPU trước khi hoàn tất.

Nguồn tự động là [WinGet](https://learn.microsoft.com/windows/package-manager/winget/),
bootstrapper [Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)
và package `Nvidia.CUDA` 12.8. Các installer chạy silent nhưng Windows vẫn hiện UAC;
đây là bước xác nhận quyền quản trị không thể và không nên bỏ qua.

| Dòng GPU | CUDA target | Trạng thái Windows | Profile |
|---|---|---|---|
| RTX 20 | `sm_75` | Hỗ trợ | Theo VRAM |
| RTX 30 | `sm_86` | Hỗ trợ | Theo VRAM |
| RTX 40 | `sm_89` | Hỗ trợ | Theo VRAM |
| RTX 50 | `sm_120` | Thử nghiệm, cần CUDA 12.8 | Chỉ `minimal` |

`auto` chọn `minimal` từ 6 GiB, `balanced` từ 8 GiB và `maximum` từ 22 GiB.
RTX 50 luôn bị khóa ở `minimal` cho đến khi có báo cáo nghiệm thu phần cứng đầy đủ.

## Cài đặt

Mở PowerShell thường và chạy một dòng. Lệnh tải bootstrap về file tạm rồi thực thi;
không dùng `Invoke-Expression`:

```powershell
$p=Join-Path $env:TEMP "thuyetminh-bootstrap.ps1"; Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/ngucungcode/thuyet-minh-offline-gpu/main/windows/bootstrap.ps1" -OutFile $p; powershell.exe -NoProfile -ExecutionPolicy Bypass -File $p
```

Bootstrap tải source vào `%LOCALAPPDATA%\Programs\ThuyetMinhOfflineGPU\source`, yêu
cầu UAC một lần, tự chọn profile theo VRAM, cài model, khởi động API + worker, chờ health
check và mở `http://127.0.0.1:8080/`. Nếu đã clone repository thì chỉ cần chạy
`.\windows\bootstrap.ps1` trong thư mục dự án.

Installer thực hiện các bước fail-closed sau:

1. Cài/repair prerequisite còn thiếu và làm mới `PATH` ngay trong tiến trình hiện tại.
2. Kiểm tra Windows build, GPU, driver, VRAM và CUDA target thật.
3. Build `llama.cpp` CUDA từ commit đã khóa riêng cho kiến trúc card hiện tại.
4. Lấy TIGER và VieNeu đúng commit, rồi xác minh các file overlay bằng SHA-256.
5. Tạo `.venv-windows`, cài PyTorch 2.8.0 `cu128` và dependency đã khóa.
6. Chạy kernel FP16, CTranslate2, worker preflight, cài model và khởi động stack.

Nếu Visual Studio hoặc NVIDIA yêu cầu reboot, installer dừng trước bước build. Khởi
động lại Windows rồi chạy lại đúng lệnh trên; các bước đã xong được nhận diện và bỏ qua,
không cài trùng.

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

# Không tự cài prerequisite và không tự khởi động (chế độ quản trị nâng cao)
.\windows\install.ps1 -Profile auto -SkipPrerequisites -SkipStart

# Dùng bootstrap nhưng không tự mở trình duyệt
.\windows\bootstrap.ps1 -NoOpenDashboard
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

- Installer yêu cầu reboot: khởi động lại Windows rồi chạy lại chính lệnh bootstrap;
  cache và prerequisite đã cài sẽ được dùng lại.
- WinGet lỗi: chạy `winget --info`; installer dùng quy trình
  `Repair-WinGetPackageManager -Force -Latest` chính thức của Microsoft khi thiếu WinGet.
- Không tìm thấy Visual Studio sau auto-install: xem log `dd_*` mới nhất trong `%TEMP%`;
  workload bắt buộc là `Microsoft.VisualStudio.Workload.VCTools`.
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
