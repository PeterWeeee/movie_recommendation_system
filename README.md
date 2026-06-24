# Hệ Thống Gợi Ý Phim Dựa Trên Lọc Cộng Tác (MovieLens 100k)

Đề tài nghiên cứu, cài đặt các thuật toán cốt lõi của Hệ thống gợi ý (Recommendation System) từ đầu và xây dựng giao diện ứng dụng trực quan hóa trong môn học Trí tuệ nhân tạo.

## 1. Cấu Trúc Thư Mục Dự Án

```text
movie_recommendation_system/
│
├── data/
│   ├── raw/                   # Thư mục chứa dữ liệu gốc từ MovieLens 100k
│   │   ├── u.data             # Tệp chứa 100,000 bản ghi điểm đánh giá
│   │   ├── u.item             # Tệp chứa thông tin danh mục tên phim
│   │   └── README             # Tài liệu hướng dẫn gốc của bộ dữ liệu
│   │
│   └── processed/             # Lưu trữ ma trận Train/Test dưới dạng nhị phân (.npy)
│                              # Giúp hệ thống nạp dữ liệu tức thì, bỏ qua bước đọc file text thô
│
├── docs/                    # Thư mục chứa tài liệu mô tả hệ thống
│   ├── production_architecture.md
│   └── system_principles.md
│
├── models/                    # Lưu trữ các file trọng số (.pkl) của tất cả các mô hình sau khi huấn luyện
│
├── notebooks/                 # Không gian nghiên cứu và trực quan hóa dữ liệu thô
│   ├── 01_data_exploration.ipynb      # Xuất báo cáo đồ thị EDA độ phân giải cao
│   ├── 02_library_verification.ipynb  # Đối chứng kết quả MAE/RMSE với thư viện scikit-surprise
│   └── 03_inspect_matrix.ipynb        # Công cụ tương tác xem ma trận dữ liệu Train/Test
│
├── scripts/                   # Các script thực thi Pipeline
│   ├── train_pipeline.py      # Tiền xử lý dữ liệu, huấn luyện mô hình (SVD, CF, Content-Based) và lưu file .pkl
│   └── ingest_to_sqlserver.py # Chuyển đổi dữ liệu và đồng bộ vào SQL Server
│
├── src/                       # Thư mục mã nguồn giải thuật cốt lõi
│   ├── __init__.py
│   ├── data_loader.py         # Đọc dữ liệu thô, chia tập Train/Test và nạp/lưu ma trận dạng numpy
│   ├── similarity.py          # Cài đặt Cosine, Pearson và Adjusted Cosine Similarity
│   ├── recommender.py         # Lớp thuật toán User-Based CF, Item-Based CF và Matrix Factorization (SVD)
│   ├── content_based.py       # Hệ thống gợi ý Content-Based dựa trên TF-IDF
│   └── evaluation.py          # Trình đo lường sai số hệ thống qua chỉ số toán học MAE và RMSE
│
├── tests/                     # Thư mục chứa các tệp Unit Test bằng pytest
│   ├── test_recommender.py    # Kiểm thử tính đúng đắn của các thuật toán gợi ý
│   └── test_similarity.py     # Kiểm thử các hàm tính độ tương đồng
│
├── app.py                     # Giao diện Web tương tác thông minh (Streamlit), dùng Hybrid Architecture
├── main.py                    # Ứng dụng gợi ý Terminal-based siêu tốc
├── requirements.txt           # Danh sách quản lý các thư viện phụ thuộc của dự án
└── .gitignore                 # Tệp cấu hình Git để loại bỏ các tệp tin rác và dữ liệu nặng khi lên GitHub
```

## 2. Yêu Cầu Môi Trường

- **Python**: 3.9 trở lên (khuyến nghị 3.11+)
- **Hệ điều hành**: Windows / macOS / Linux

## 3. Hướng Dẫn Cài Đặt & Chạy

### Bước 1: Clone repository

```bash
git clone https://github.com/PeterWeeee/movie_recommendation_system.git
cd movie_recommendation_system
```

### Bước 2: Tạo và kích hoạt môi trường ảo

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS/Linux)
source .venv/bin/activate
```

### Bước 3: Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị dữ liệu

Tải bộ dữ liệu **MovieLens 100k** từ [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/) và đặt các file sau vào thư mục `data/raw/`:

```
data/raw/u.data
data/raw/u.item
```

### Bước 5: Huấn luyện Mô hình (Bắt buộc chạy lần đầu)

Chạy pipeline để hệ thống tự động xử lý dữ liệu và xuất các mô hình học máy ra thư mục `models/`:

```bash
python scripts/train_pipeline.py
```

### Bước 6: Chạy ứng dụng

**Chạy giao diện Web (Streamlit):**

```bash
streamlit run app.py
```

Truy cập trình duyệt tại: `http://localhost:8501`

**Chạy kiểm thử qua Terminal:**

```bash
python main.py
```

**Chạy Unit Test:**

```bash
python -m pytest tests/ -v
```

## 4. Các Thuật Toán Được Cài Đặt

| Thuật toán | Mô tả |
|---|---|
| **Global Baseline (Biased Predictor)** | Dự đoán điểm số dựa trên xu hướng toàn cục: $r = \mu + b_u + b_i$, cung cấp mốc tham chiếu cho các mô hình phức tạp. |
| **User-Based Collaborative Filtering** | Hỗ trợ 3 chế độ: Basic, Means (KNN with Means) và Biased Baseline. Dùng Pearson Similarity tìm người dùng tương đồng. |
| **Item-Based Collaborative Filtering** | Hỗ trợ chế độ Basic và Biased Baseline. Đánh giá độ tương đồng giữa các phim bằng Cosine / Adjusted Cosine Similarity. |
| **Matrix Factorization (SVD + SGD)** | Phân rã ma trận User-Item thành các nhân tử ẩn (latent factors), huấn luyện bằng Stochastic Gradient Descent với L2 Regularization |
| **Content-Based Filtering** | Phân tích thể loại phim bằng TF-IDF và tính khoảng cách Cosine để tìm phim tương đồng về nội dung |

## 5. Kết Quả Đánh Giá (MovieLens 100k — Test Set 20%)

| Thuật toán | RMSE | MAE |
|---|---|---|
| User-Based CF (KNN Basic) | 1.0934 | 0.8496 |
| User-Based CF (KNN Means) | 0.9368 | 0.7323 |
| User-Based CF (KNN Biased Baseline) | 0.9261 | 0.7262 |
| Item-Based CF (KNN Basic) | 1.2336 | 0.9230 |
| Item-Based CF (KNN Biased Baseline) | 0.9145 | 0.7188 |
| Matrix Factorization SVD | 0.9362 | 0.7407 |

> Đối chiếu tham khảo từ thư viện `scikit-surprise` (notebook `02_library_verification.ipynb`):
> KNNWithMeans User-Based: RMSE 0.9197 / MAE 0.7190 — KNNWithMeans Item-Based: RMSE 0.9169 / MAE 0.7188

## 6. Chức Năng Giao Diện

- 🔑 **Thanh Điều Hướng & Đăng Nhập**: Hỗ trợ đăng nhập mô phỏng bằng User ID, tạo User mới với các thông tin nhân khẩu học cơ bản và cấu hình TMDB API Key.
- 🏠 **Trang Chủ (Khám Phá)**: Trải nghiệm ứng dụng thực tế với danh sách phim "Dành Cho Bạn" (cá nhân hóa bằng SVD), "Đang Thịnh Hành" (dựa trên chất lượng nội tại từ Biased Predictor), và tìm kiếm "Phim Tương Tự" (Item-Based CF). Tích hợp TMDB API để hiển thị ảnh bìa phim trực quan, sinh động.
- 👤 **Đánh giá của người dùng**: Quản lý hồ sơ cá nhân với 2 tab:
  - **Lịch sử đánh giá**: Xem lại danh sách các phim đã đánh giá dưới dạng thẻ ảnh trực quan hoặc dạng bảng.
  - **Đánh giá film**: Tìm kiếm, chấm điểm phim mới và cập nhật trực tiếp vào hệ thống (cập nhật qua ma trận/SQL).
- 👨‍💻 **Dành Cho Developer**: Không gian phân tích kỹ thuật chuyên sâu với 3 tab:
  - **Trực Quan Hóa Dữ Liệu**: Biểu đồ độ thưa thớt (Sparsity), hiện tượng Long-Tail, và không gian đặc trưng ẩn 2D của SVD (PCA).
  - **Phân tích và Đánh giá Mô hình**: Đồ thị Training Loss của SVD và bảng so sánh MAE, RMSE, Precision@K, Recall@K, và Tốc độ dự đoán giữa các thuật toán. Bảng so sánh nay cung cấp cái nhìn đối sánh trực tiếp giữa các chuẩn khoảng cách: User-Based (Pearson vs Cosine), Item-Based (Adjusted Cosine vs Cosine), và SVD.
  - **So Sánh Thuật Toán**: So sánh song song top phim gợi ý từ các thuật toán khác nhau cho cùng một User, kèm theo giải thích chi tiết các bước thuật toán đã làm khi gợi ý phim:
    - **User-Based CF**: Trích xuất dữ liệu nền, tìm láng giềng tương đồng (những người dùng giống nhau), điều chỉnh sai số và tổng hợp thành dự đoán.
    - **Item-Based CF**: Đánh giá dựa trên độ tương đồng giữa các bộ phim, trích xuất ma trận con, và tính toán dự đoán từ các phim láng giềng.
    - **SVD (Matrix Factorization)**: Tách các thành phần mốc nền (Bias), khớp không gian đặc trưng ẩn (Latent Factors) giữa User và Item, rồi dùng công thức tích vô hướng để đưa ra điểm dự đoán cuối cùng.

## 7. Thư Viện Sử Dụng

Xem chi tiết tại [`requirements.txt`](requirements.txt):

- `numpy` — Tính toán ma trận và đại số tuyến tính
- `pandas` — Đọc và xử lý dữ liệu dạng bảng
- `streamlit` — Xây dựng giao diện Web tương tác
- `matplotlib` — Vẽ biểu đồ trực quan hóa
- `seaborn` — Biểu đồ thống kê nâng cao
- `scikit-learn` — PCA để trực quan hóa latent features của SVD
- `scikit-surprise` — Thư viện đối chứng kết quả thuật toán (notebook verification)

## 8. Phân Tích Kỹ Thuật

- **Cold Start (Khởi động lạnh)**:
  - Hệ thống gặp khó khăn khi gợi ý cho người dùng mới (chưa có đánh giá nào) hoặc phim mới (chưa ai đánh giá).
  - *Giải pháp*: Sử dụng Content-Based Filtering cho phim mới (dựa vào thể loại, đạo diễn), hoặc dùng Popularity-based/Global Baseline để gợi ý cho người dùng mới. Hiện tại hệ thống kết hợp gợi ý "Đang Thịnh Hành" cho user mới.

- **Sparsity (Độ thưa thớt dữ liệu)**:
  - Ma trận User-Item trong thực tế thường có độ thưa thớt rất cao (trên 90%). Trong MovieLens 100k, độ thưa thớt khoảng 93.7%.
  - *Giải pháp*: Thuật toán Matrix Factorization (SVD) thể hiện ưu điểm vượt trội so với Memory-based CF trong việc xử lý ma trận thưa do khả năng xấp xỉ các giá trị bị khuyết thông qua latent features.

- **Scalability (Khả năng mở rộng)**:
  - Thuật toán Memory-based (User/Item CF) gặp hạn chế về bộ nhớ và thời gian tính toán khi số lượng người dùng/phim tăng lên mức hàng triệu do phải tính toán và lưu trữ ma trận độ tương đồng kích thước lớn ($O(N^2)$ hoặc $O(M^2)$).
  - *Giải pháp*: Sử dụng Model-based CF (SVD) giúp việc dự đoán ở thời gian thực nhanh chóng do chỉ cần tính tích vô hướng của 2 vector latent nhỏ gọn. Bước huấn luyện tốn kém có thể chạy offline định kỳ.