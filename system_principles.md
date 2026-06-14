# Nguyên Lý Hoạt Động Của Hệ Thống Gợi Ý Phim (MovieLens 100k)

Tài liệu này giải thích chi tiết các nền tảng toán học và quy trình kỹ thuật đằng sau hệ thống gợi ý phim. Hệ thống kết hợp nhiều phương pháp tiếp cận từ truyền thống đến máy học nâng cao.

---

## 1. Dữ Liệu Đầu Vào & Tiền Xử Lý
Hệ thống sử dụng bộ dữ liệu **MovieLens 100k**, bao gồm 100.000 lượt đánh giá (từ 1 đến 5 sao) của 943 người dùng cho 1682 bộ phim.
- **Tiền xử lý (Data Loader)**: File `u.data` thô được đọc và ánh xạ thành một ma trận 2 chiều `User-Item Matrix` (kích thước $943 \times 1682$).
- **Tối ưu tốc độ**: Các ma trận Train (80%) và Test (20%) sau khi chia tách sẽ được lưu dưới dạng file nhị phân của Numpy (`.npy`). Nhờ vậy, ở những lần chạy sau, hệ thống không phải tốn thời gian phân tích cú pháp từ file text thô.

---

## 2. Các Thuật Toán Cốt Lõi

Hệ thống triển khai 3 phân nhánh cốt lõi của bài toán Hệ thống gợi ý:

### 2.1. Lọc Dựa Trên Nội Dung (Content-Based Filtering)
- **Mục tiêu**: Gợi ý các bộ phim có thể loại hoặc chủ đề tương đồng với phim người dùng đang quan tâm.
- **Cách hoạt động**: Hệ thống nối tên phim và các nhãn thể loại (Action, Romance, Sci-Fi...) thành một chuỗi văn bản. Sau đó, sử dụng **TF-IDF Vectorizer** để số hóa và chấm điểm trọng số các từ khóa. Cuối cùng, tính **Cosine Similarity** giữa các vector để tìm ra phim giống nhau nhất.
- **Vai trò trong hệ thống**: Dùng để hỗ trợ giải quyết bài toán "Khởi động lạnh" (Cold Start) khi người dùng hoàn toàn mới chưa có lịch sử đánh giá.

### 2.2. Lọc Cộng Tác Dựa Trên Bộ Nhớ (Memory-Based CF / KNN)
Thuật toán dựa trên nguyên lý: *Những người từng có chung sở thích trong quá khứ sẽ tiếp tục thích những bộ phim giống nhau trong tương lai.*
- **Tính độ tương đồng (Pearson Correlation)**: Để tìm K người "hàng xóm" gần nhất (KNN), hệ thống tính hệ số Pearson giữa các User. Hệ số này đặc biệt hiệu quả vì nó trừ đi điểm đánh giá trung bình của từng người, qua đó "chuẩn hóa" được thói quen của những người hay cho điểm quá cao hoặc quá thấp.
- **Dự đoán điểm (KNN with Biased Baseline)**: Thay vì chỉ lấy trung bình điểm của K người hàng xóm, hệ thống sử dụng kết hợp **Biased Predictor** (Tính toán độ chênh lệch của bản thân User đó và độ phổ biến của bộ phim) để đưa ra dự đoán chính xác hơn.

### 2.3. Lọc Cộng Tác Dựa Trên Mô Hình (Matrix Factorization - SVD)
Được coi là thuật toán mạnh mẽ nhất cho hệ thống gợi ý, SVD giải quyết vấn đề "Ma trận thưa thớt" (Sparsity) - nơi 95% các ô trong ma trận không có điểm đánh giá.
- **Nguyên lý phân rã**: Ma trận $R$ (Users x Items) khổng lồ được tách thành 2 ma trận nhỏ hơn đại diện cho các đặc trưng ẩn (Latent Factors): $P$ (User) và $Q$ (Item).
- **Học máy (Stochastic Gradient Descent - SGD)**: Hệ thống lặp qua nhiều Epochs để cập nhật $P$ và $Q$, nhằm giảm thiểu sai số giữa điểm thực tế và điểm dự đoán.
- **L2 Regularization ($\lambda$)**: Cả trọng số ẩn và các hệ số Bias đều bị phạt (penalize) nếu có giá trị quá lớn, qua đó tránh được hiện tượng học vẹt (Overfitting).

### 2.4. Hướng Phát Triển Tương Lai (SVD++)
- **Nâng cấp từ SVD**: Mặc dù SVD đã rất mạnh mẽ, nhưng nó chỉ sử dụng các đánh giá **rõ ràng (explicit feedback)** (điểm số từ 1 đến 5).
- **Thuật toán SVD++**: Trong tương lai, dự án hướng tới việc tích hợp SVD++, một thuật toán tiên tiến hơn bằng cách đưa thêm các tín hiệu **ngầm (implicit feedback)** vào mô hình.
- **Nguyên lý**: Việc một người dùng "chọn xem" hay "nhấp chuột" vào một bộ phim (dù họ đánh giá cao hay thấp) cũng chứa đựng thông tin về sở thích của họ. SVD++ bổ sung thêm một tập hợp các latent factors đại diện cho các item mà user đã tương tác, giúp tăng cường độ chính xác đáng kể, đặc biệt cho những user có ít đánh giá.

---

## 3. Quy Trình Phục Vụ (Inference Pipeline)

Khi người dùng (Ví dụ: `User ID = 1`) truy cập hệ thống và yêu cầu gợi ý Top 5 bộ phim:

1. **Kiểm tra Cold Start**:
   - Nếu `User ID` không tồn tại trong tập huấn luyện (Người dùng mới), hệ thống sẽ không thể áp dụng thuật toán SVD hay KNN do thiếu dữ liệu. Hệ thống tự động kích hoạt **Popularity-Based** (Gợi ý phim phổ biến nhất toàn sàn).
2. **Lọc phim đã xem**:
   - Hệ thống đối chiếu hàng `User ID = 1` trong ma trận Train. Mọi bộ phim đã có điểm đánh giá (khác 0) sẽ bị loại bỏ khỏi danh sách cân nhắc.
3. **Chấm điểm dự kiến (Predict Rating)**:
   - Các bộ phim chưa xem còn lại sẽ được đưa qua mô hình học máy (ví dụ: SVD). Mô hình tính tích vô hướng (Dot Product) giữa vector đặc trưng của User 1 và vector đặc trưng của bộ phim, cộng với hệ số Bias để sinh ra điểm dự kiến (từ 1 đến 5 sao).
4. **Sắp xếp và Trả về**:
   - Danh sách được sắp xếp giảm dần theo điểm dự kiến. Top 5 bộ phim cao điểm nhất được lấy ra, truy vấn ngược lại file `u.item` để lấy tên phim và hiển thị ra giao diện Web.

---

## 4. Đo Lường & Đánh Giá

Để chứng minh tính hiệu quả của các công thức tự xây dựng từ đầu (Scratch), hệ thống liên tục tính toán trên tập Test (20% dữ liệu ẩn):
- **MAE (Mean Absolute Error)**: Độ lệch tuyệt đối trung bình (Ví dụ: Dự đoán 4 sao, thực tế 3.5 sao -> MAE = 0.5).
- **RMSE (Root Mean Square Error)**: Phạt nặng hơn đối với các dự đoán sai lệch quá lớn.
Hệ thống cũng được so sánh ngang hàng (Benchmark) với thư viện chuyên dụng `scikit-surprise` để đảm bảo kết quả tính toán có tính thực tiễn và chuẩn xác theo chuẩn khoa học dữ liệu.
