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
├── models/                    # Lưu trữ các file trọng số (.pkl) của tất cả các mô hình sau khi huấn luyện
│
├── notebooks/                 # Không gian nghiên cứu và trực quan hóa dữ liệu thô
│   ├── 01_data_exploration.ipynb      # Xuất báo cáo đồ thị EDA độ phân giải cao
│   └── 02_library_verification.ipynb # Đối chứng kết quả MAE/RMSE với thư viện scikit-surprise
│
├── scripts/                   # Các script thực thi Pipeline
│   └── train_pipeline.py      # Tiền xử lý dữ liệu, huấn luyện mô hình (SVD, CF, Content-Based) và lưu file .pkl
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
├── production_architecture.md # Tài liệu mô tả kiến trúc hệ thống production
├── system_principles.md       # Tài liệu mô tả nguyên lý hoạt động các thuật toán
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
| **User-Based Collaborative Filtering** | Tìm K người dùng có sở thích tương đồng nhất (Pearson Similarity) và dự đoán điểm số theo công thức KNN with Biased Baseline |
| **Item-Based Collaborative Filtering** | Đánh giá độ tương đồng giữa các phim (Adjusted Cosine Similarity), dự đoán theo KNN with Biased Baseline |
| **Matrix Factorization (SVD + SGD)** | Phân rã ma trận User-Item thành các nhân tử ẩn (latent factors), huấn luyện bằng Stochastic Gradient Descent với L2 Regularization |
| **Content-Based Filtering** | Phân tích thể loại phim bằng TF-IDF và tính khoảng cách Cosine để tìm phim tương đồng về nội dung |

## 5. Kết Quả Đánh Giá (MovieLens 100k — Test Set 20%)

| Thuật toán | RMSE | MAE |
|---|---|---|
| User-Based CF (KNN Biased Baseline) | 0.9261 | 0.7262 |
| Item-Based CF (KNN Biased Baseline) | 0.9146 | 0.7189 |
| Matrix Factorization SVD | 0.9362 | 0.7407 |

> Đối chiếu tham khảo từ thư viện `scikit-surprise` (notebook `02_library_verification.ipynb`):
> KNNBaseline User-Based: RMSE 0.9197 / MAE 0.7190 — KNNBaseline Item-Based: RMSE 0.9169 / MAE 0.7188

## 6. Chức Năng Giao Diện

- 🏠 **Trang Chủ (Khám Phá)**: Trải nghiệm ứng dụng thực tế với danh sách phim "Dành Cho Bạn" (cá nhân hóa bằng SVD), "Đang Thịnh Hành" (dựa trên chất lượng nội tại từ Biased Predictor), và tìm kiếm "Phim Tương Tự" (Item-Based CF). Tích hợp TMDB API để hiển thị ảnh bìa phim trực quan, sinh động.
- ⚙️ **Dành Cho Developer**: Không gian phân tích kỹ thuật chuyên sâu với 3 tab:
  - 📊 **Trực Quan Hóa Dữ Liệu**: Biểu đồ độ thưa thớt (Sparsity), hiện tượng Long-Tail, và không gian đặc trưng ẩn 2D của SVD (PCA).
  - 📈 **Đánh Giá Hiệu Năng**: Đồ thị Training Loss của SVD và bảng so sánh MAE/RMSE giữa các thuật toán.
  - 🧪 **So Sánh Thuật Toán**: So sánh song song top phim gợi ý từ các thuật toán khác nhau (User-Based, Item-Based, SVD) cho cùng một User.

## 7. Thư Viện Sử Dụng

Xem chi tiết tại [`requirements.txt`](requirements.txt):

- `numpy` — Tính toán ma trận và đại số tuyến tính
- `pandas` — Đọc và xử lý dữ liệu dạng bảng
- `streamlit` — Xây dựng giao diện Web tương tác
- `matplotlib` — Vẽ biểu đồ trực quan hóa
- `seaborn` — Biểu đồ thống kê nâng cao
- `scikit-learn` — PCA để trực quan hóa latent features của SVD
- `scikit-surprise` — Thư viện đối chứng kết quả thuật toán (notebook verification)