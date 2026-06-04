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
├── models/                    # Đóng gói trọng số mô hình SVD (svd_weights.pkl) sau khi học
│                              # Phục vụ chạy Demo giao diện ngay lập tức không cần huấn luyện lại
│
├── notebooks/                 # Không gian nghiên cứu và trực quan hóa dữ liệu thô
│   └── 01_data_exploration.ipynb  # File Jupyter Notebook xuất báo cáo đồ thị độ phân giải cao
│
├── src/                       # Thư mục mã nguồn giải thuật cốt lõi
│   ├── __init__.py
│   ├── data_loader.py         # Đọc dữ liệu, chia tập Train/Test độc lập và quản lý dữ liệu processed
│   ├── similarity.py          # Cài đặt thuần toán học Cosine và Pearson Similarity
│   ├── recommender.py         # Lớp thuật toán User-Based CF (KNN) và Matrix Factorization (SVD) với SGD
│   └── evaluation.py          # Trình đo lường sai số hệ thống qua chỉ số toán học MAE và RMSE
│
├── app.py                     # Mã nguồn thiết kế giao diện Web tương tác thông minh (Streamlit)
├── main.py                    # File thực thi kiểm thử hệ thống qua Terminal
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

### Bước 5: Chạy ứng dụng

**Chạy giao diện Web (Streamlit):**

```bash
streamlit run app.py
```

Truy cập trình duyệt tại: `http://localhost:8501`

**Chạy kiểm thử qua Terminal:**

```bash
python main.py
```

## 4. Các Thuật Toán Được Cài Đặt

| Thuật toán | Mô tả |
|---|---|
| **User-Based Collaborative Filtering (KNN)** | Tìm K người dùng có sở thích tương đồng nhất (Pearson Similarity) và dự đoán điểm số |
| **Matrix Factorization (SVD + SGD)** | Phân rã ma trận User-Item thành các nhân tử ẩn, huấn luyện bằng Stochastic Gradient Descent |

## 5. Chức Năng Giao Diện

- 📊 **Tab 1 — Trực Quan Hóa**: So sánh phân phối dữ liệu trước và sau xử lý, heatmap ma trận tương đồng
- 🎯 **Tab 2 — Gợi Ý Phim**: Nhập User ID, chọn thuật toán, nhận Top-N phim được gợi ý
- 📈 **Tab 3 — Đánh Giá Hiệu Năng**: So sánh chỉ số MAE và RMSE giữa các thuật toán trên tập Test

## 6. Thư Viện Sử Dụng

Xem chi tiết tại [`requirements.txt`](requirements.txt):

- `numpy` — Tính toán ma trận và đại số tuyến tính
- `pandas` — Đọc và xử lý dữ liệu dạng bảng
- `streamlit` — Xây dựng giao diện Web tương tác
- `matplotlib` — Vẽ biểu đồ trực quan hóa
- `seaborn` — Biểu đồ thống kê nâng cao