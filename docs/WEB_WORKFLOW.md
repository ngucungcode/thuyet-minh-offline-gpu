# Workflow web và tích hợp nguồn

## Luồng làm một bản thuyết minh

1. Mở `http://127.0.0.1:8080/` qua SSH tunnel và kiểm tra thẻ GPU/acquisition.
2. Tìm tên phim, có thể thêm năm; chọn đúng release hoặc dán `Release ID` từ kết
   quả tìm kiếm.
3. Chọn ngôn ngữ nguồn, chế độ phụ đề và các model đã cài/verify. Nếu dùng giọng
   tham chiếu, chỉ chọn tệp giọng mà bạn có quyền sử dụng và xác nhận quyền giọng.
4. Xác nhận quyền đối với nội dung rồi tạo job.
5. Nếu job dừng ở `needs_subtitle_selection`, chọn một ứng viên hoặc chọn dùng
   ASR. Nếu dừng ở `needs_language`, chọn ngôn ngữ đúng để pipeline tiếp tục.
6. Theo dõi các stage acquisition → subtitle/ASR → translation → separation →
   TTS → timing → mix → export → verify.
7. Khi hoàn tất, tải MP4, phụ đề tiếng Việt SRT và timing report JSON. Có thể hủy
   job đang chạy hoặc tiếp tục job đã pause/lỗi có thể retry ngay trên web.

## Thêm indexer vào Prowlarr

Prowlarr sở hữu schema và credential riêng cho từng provider, nên dashboard
không tự dựng một biểu mẫu indexer chung. Tham khảo
[tài liệu/API Prowlarr](https://prowlarr.com/docs/api/); cách cấu hình:

1. Tạo tunnel cả API và Prowlarr:

   ```bash
   ssh -p SSH_PORT \
     -L 8080:127.0.0.1:8080 \
     -L 9696:127.0.0.1:9696 \
     root@GPU_HOST
   ```

2. Mở `http://127.0.0.1:9696`, vào **Indexers → Add Indexer**.
3. Chọn indexer mà tài khoản/khu vực của bạn cho phép, nhập credential, bấm
   **Test**, rồi **Save**.
4. Trở lại mục **Tích hợp** của dashboard, bấm **Kiểm thử tất cả**. Indexer hợp
   lệ sẽ được dùng trong ô tìm phim; không cần sao chép API key nội bộ của
   Prowlarr vào trình duyệt.

Chỉ cấu hình nguồn mà bạn được phép truy cập và chỉ tải nội dung bạn sở hữu hoặc
được cấp quyền. Dự án không cài sẵn tracker cụ thể.

## Kết nối OpenSubtitles API

OpenSubtitles yêu cầu các request sau đăng nhập tiếp tục dùng `base_url` trả về;
dashboard lưu route này sau khi giới hạn nó vào các host API chính thức. Xem
[tài liệu API OpenSubtitles](https://ai.opensubtitles.com/docs).

1. Đăng ký/đăng nhập OpenSubtitles và tạo API consumer key.
2. Trong mục **Tích hợp**, nhập API key, username và password rồi bấm lưu.
3. Backend gọi endpoint login chính thức, kiểm tra tài khoản trên đúng `base_url`
   được trả về rồi ghi atomic một bundle gồm `opensubtitles_api_key`,
   `opensubtitles_token` và `opensubtitles_base_url` với mode `0600`. Password chỉ
   tồn tại trong request đăng nhập và không được ghi xuống đĩa hoặc trả lại.
4. Chạy `dub stack restart`; sau đó mở lại dashboard để kiểm tra trạng thái.
5. Nếu token hết hạn, thực hiện đăng nhập lại. Nút xóa cấu hình xóa cả key, token
   và route API; nếu dọn file bị gián đoạn, dashboard hiện nút retry riêng.

Luồng ghi secret từ web dành cho deployment native, nơi `var/secrets` thuộc user
chạy API. Docker Compose mount `/run/secrets` chỉ đọc; ở profile đó hãy sửa file
secret trên host, đặt `DUB_OPENSUBTITLES_URL` trong `.env` đúng bằng `base_url`
trả về khi login, rồi chạy `docker compose up -d --force-recreate api worker`.

Quản trị tích hợp chỉ hoạt động từ loopback và header riêng của dashboard. Giao
diện production được FastAPI phục vụ cùng origin với API; không cần một web
server công khai khác.
