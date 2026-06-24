# Kịch Bản Kiểm Tra Hệ Thống Gợi Ý Lọc Cộng Tác (Collaborative Filtering) từ Scratch

Tài liệu này hướng dẫn kiểm tra tính chính xác của thuật toán gợi ý phim được xây dựng từ đầu (scratch) trên bộ dữ liệu MovieLens 100k (bao gồm các file `u.data`, `u.item`, v.v.).

---

## 1. Kiểm Tra Tính Đúng Đắn Của Tiền Xử Lý Dữ Liệu (Data Preprocessing)

### Đọc dữ liệu và Ma trận Bản ghi (User-Item Matrix)
* **Yêu cầu:** Đọc file `u.data` (User id | Item id | Rating | Timestamp) và chuyển đổi thành ma trận $R$ kích thước $M \times N$ (với $M = 943$ người dùng và $N = 1682$ bộ phim).
* **Cần kiểm tra:**
    * Kích thước ma trận chính xác là $(943, 1682)$.
    * Các giá trị thiếu phải được gán bằng `0` hoặc `NaN`.
    * Giá trị rating phải nằm trong khoảng $[1, 5]$. Kiểm tra ngẫu nhiên một vài tọa độ $(u, i)$ xem giá trị có khớp với file gốc hay không.

### Phân chia dữ liệu (Train/Test Split)
* **Yêu cầu:** Kiểm tra việc chia dữ liệu (ví dụ: 80% train, 20% test).
* **Cần kiểm tra:**
    * Tổng số lượng rating trong tập Train + Test phải bằng đúng số rating ban đầu (100,000).
    * Đảm bảo không bị rò rỉ dữ liệu (Data Leakage) từ tập Test sang tập Train.

---

## 2. Kiểm Tra Thuật Toán Tính Độ Tương Đồng (Similarity Matrix)

Kiểm tra các hàm tính toán độ tương đồng (Cosine, Pearson, hoặc Jaccard).

### Ma trận Tương đồng Người dùng (User-User) hoặc Vật phẩm (Item-Item)
* **Cần kiểm tra:**
    * Kích thước ma trận User-User phải là $(943, 943)$.
    * Kích thước ma trận Item-Item phải là $(1682, 1682)$.
    * **Tính đối xứng:** Ma trận tương đồng phải đối xứng qua đường chéo chính ($S_{i,j} = S_{j,i}$).
    * **Đường chéo chính:** Tất cả các giá trị trên đường chéo chính phải bằng $1.0$ (hoặc gần bằng $1.0$ do sai số dấu phẩy động).
    * **Khoảng giá trị:** Tất cả các giá trị độ tương đồng phải nằm trong khoảng $[-1, 1]$ (đối với Cosine/Pearson).

---

## 3. Kiểm Tra Hàm Dự Đoán Điểm Số (Rating Prediction)

### Dự đoán không chuẩn hóa và có chuẩn hóa (Mean-Centering)
* **Yêu cầu:** Kiểm tra công thức dự đoán điểm số $\hat{r}_{u,i}$ cho một người dùng $u$ và phim $i$ chưa xem.
* **Cần kiểm tra:**
    * **Giới hạn biên:** Điểm số dự đoán $\hat{r}_{u,i}$ không được vượt quá khoảng $[1, 5]$. Nếu vượt quá hoặc nhỏ hơn, thuật toán phải có cơ chế kẹp giá trị (clipping).
    * **Trường hợp đặc biệt:** Nếu một bộ phim chưa có ai đánh giá, hoặc một người dùng chưa đánh giá phim nào (Cold Start), thuật toán có bị lỗi chia cho 0 (`NaN`) hay không? Hệ thống phải xử lý bằng cách trả về điểm trung bình toàn cục (Global Mean) hoặc điểm trung bình của người dùng/vật phẩm đó.
    * **Kiểm tra thủ công (Sanity Check):** Chọn 1 người dùng và 1 bộ phim, lấy danh sách $K$ láng giềng gần nhất (K-NN), tự tính toán bằng tay hoặc bằng thư viện chuẩn (như `scipy` hoặc `scikit-learn`) để đối chiếu kết quả của hàm tự viết.

---

## 4. Kiểm Tra Thuật Toán Matrix Factorization (Nếu có sử dụng SVD/SGD)

* **Cần kiểm tra:**
    * Kích thước các ma trận nhân tử: Nếu số chiều ẩn là $K$, ma trận User $P$ phải có kích thước $(943, K)$ và ma trận Item $Q$ phải có kích thước $(1682, K)$.
    * **Hàm mất mát (Loss Function):** Giá trị Loss (MSE hoặc RMSE) trên tập Train phải giảm dần sau mỗi vòng lặp (epoch). Nếu Loss tăng hoặc dao động mạnh, cần kiểm tra lại tốc độ học (learning rate) và đạo hàm của hàm cập nhật.

---

## 5. Đánh Giá Hiệu Năng Trên Tập Test (Evaluation Metrics)

Hệ thống phải tính toán chính xác các chỉ số đánh giá sau trên tập dữ liệu Test:

### Chỉ số dự đoán (Rating Metrics)
* **MAE (Mean Absolute Error):**
    $$MAE = \frac{1}{|\Omega_{test}|} \sum_{(u,i) \in \Omega_{test}} |r_{u,i} - \hat{r}_{u,i}|$$
* **RMSE (Root Mean Squared Error):**
    $$RMSE = \sqrt{\frac{1}{|\Omega_{test}|} \sum_{(u,i) \in \Omega_{test}} (r_{u,i} - \hat{r}_{u,i})^2}$$
* **Yêu cầu đối chứng:** Đối với MovieLens 100k, thuật toán Collaborative Filtering chuẩn thường đạt RMSE trong khoảng từ `0.90` đến `1.05`. Nếu RMSE nhận được quá cao (> 1.5) hoặc quá thấp (< 0.5 trên tập test), thuật toán chắc chắn đã bị lỗi logic hoặc Overfitting.

### Chỉ số xếp hạng (Ranking Metrics - Top-N Recommendation)
* Nếu hệ thống có tính năng gợi ý danh sách Top-N phim:
    * Kiểm tra tính chính xác của các hàm tính **Precision@N**, **Recall@N**, và **NDCG@N**.
    * Đảm bảo danh sách gợi ý cho một người dùng không chứa những bộ phim mà người dùng đó đã đánh giá trong tập Train.