# Lồng Tiếng GPU Studio — Web

Dashboard tiếng Việt được tích hợp trực tiếp vào `thuyet-minh-offline-gpu`.
Bản production do FastAPI phục vụ cùng origin với `/v1/*`, vì vậy không cần
Node.js, reverse proxy hay cấu hình CORS trên máy GPU.

## Chức năng

- Kiểm tra trạng thái API, GPU và catalog model.
- Tìm nguồn qua indexer đã được quản trị viên cấu hình.
- Chọn kết quả hoặc nhập `Release ID`, ngôn ngữ nguồn và chế độ phụ đề.
- Bắt buộc xác nhận quyền trước khi tạo job.
- Tự cập nhật danh sách job và tiến độ từng công đoạn mỗi 3 giây.
- Hủy, tiếp tục và tải MP4 hoàn tất.
- Hiển thị tốt trên desktop và điện thoại.

API hiện tại chưa có endpoint nhập magnet hoặc upload `.torrent` trực tiếp.
Web không giả lập tính năng này: cần bổ sung hợp đồng backend trước khi bật giao
diện tương ứng.

## Chạy bản đã tích hợp

Từ thư mục gốc của dự án:

```bash
dub serve
```

Mở `http://127.0.0.1:8080/`. API vẫn ở `http://127.0.0.1:8080/v1/*`.
Installer hiện có không cần chạy thêm dịch vụ web.

Để truy cập từ máy cá nhân tới GPU VM, giữ API bind loopback và dùng SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 -p SSH_PORT root@GPU_HOST
```

Sau đó mở `http://127.0.0.1:8080/` trên máy cá nhân.

## Phát triển frontend

Yêu cầu Node.js 22.13 trở lên. Tạo `.env.local` nếu API không dùng địa chỉ mặc
định:

```dotenv
DUB_API_URL=http://127.0.0.1:8080
```

Chạy frontend ở cổng 3000 với route `/v1/*` chuyển tiếp về backend:

```bash
npm ci
npm run dev
```

## Cập nhật asset trong gói Python

Sau khi sửa frontend, chạy:

```bash
npm ci
npm run embed
```

Lệnh này build Vinext, render `index.html` và thay atomically thư mục
`src/dub_server/web_static`. Các file sinh ra được đóng gói trong wheel nhờ cấu
hình `package-data` của `pyproject.toml`; production không cần Node.js.

Không công khai cổng API, qBittorrent hoặc Prowlarr ra Internet. Dashboard này
không thay thế bước xác minh quyền sử dụng nội dung.
