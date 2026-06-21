Dưới đây là các yêu cầu cập nhật giao diện và tính năng cho file `app.py`. Vui lòng thực hiện các thay đổi sau theo từng khu vực:

**1. Khu vực Đăng nhập & User:**
* **Sửa lỗi cập nhật trạng thái:** Hiện tại, sau khi tạo user mới, user đó không xuất hiện trong danh sách lựa chọn ở phần "Đăng nhập". Hãy khắc phục lỗi này để danh sách dropdown được cập nhật ngay lập tức.
* **Tái cấu trúc UI:** Di chuyển toàn bộ form/tính năng "Tạo user mới" vào thẳng phần "Đăng nhập" (trước đây nằm ở mục "Quản lý user & rating").

**2. Trang chủ (Khám phá):**
* **Xóa UI:** Xóa bỏ phần hiển thị "Phim đã đánh giá" nằm bên trong mục "Dành cho bạn".
* **Thêm tính năng "Xem thêm" (áp dụng cho các mục: "Dành cho bạn", "Khám phá film", và "Phim đang thịnh hành"):** 
    * Hiển thị mặc định 5 bộ phim. Thêm nút "Xem thêm" bên cạnh tiêu đề của các mục này.
    * Khi nhấn "Xem thêm" lần 1: Bảng mở rộng thêm 1 hàng (5 bộ phim). Phía dưới danh sách hiện thêm nút "Tải thêm 5 bộ phim" để người dùng tiếp tục xem thêm nếu muốn.
    * Khi nhấn nút "Xem thêm" lần nữa (chuyển thành Thu gọn): Danh sách phim được rút gọn lại về trạng thái mặc định ban đầu (5 bộ phim) và ẩn nút tải thêm.

**3. Đổi tên và thiết kế lại "Quản lý user & rating":**
* Đổi tên mục này trên thanh điều hướng thành: "Đánh giá của người dùng".
* Trong trang này, chia làm 2 phần (hoặc tab):
    * **Lịch sử đánh giá:** Cung cấp tùy chọn cho phép người dùng xem dưới dạng "Danh sách rút gọn" hoặc "Danh sách mở rộng" (hiển thị kèm ảnh poster phim và điểm đánh giá).
    * **Đánh giá film:** Thêm thanh tìm kiếm theo tên phim. Mặc định khi chưa tìm kiếm, hãy hiển thị danh sách các bộ phim được gợi ý bởi thuật toán SVD.

**4. Dành cho Developer (So sánh thuật toán):**
* **Tùy chỉnh số lượng:** Thêm một thanh kéo (slider) cho phép chọn số lượng bộ phim muốn gợi ý.
* **Chi tiết so sánh:** 
    * Khi nhấn nút "Gợi ý phim", hệ thống chạy và hiện thêm một nút "Xem chi tiết so sánh".
    * Nhấn nút "Xem chi tiết so sánh" sẽ hiển thị ra 3 bảng/cột đại diện cho: `User-based CF`, `Item-based CF`, và `SVD`.
    * **Chi tiết trong mỗi bảng:**
        * Cung cấp các lựa chọn để xem từng bước thực thi của thuật toán đó.
        * Có thể chọn các biến thể của thuật toán để xem kết quả. (Ví dụ: với bảng User-based CF, có thể chọn xem theo phương pháp thuần túy, phương pháp với means, hoặc phương pháp biased baseline).
