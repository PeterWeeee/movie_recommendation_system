# Hệ Thống Gợi Ý Phim Dựa Trên Lọc Cộng Tác (MovieLens 100k)

Đề tài nghiên cứu, cài đặt các thuật toán cốt lõi của Hệ thống gợi ý (Recommendation System) từ đầu và xây dựng giao diện ứng dụng trực quan hóa trong môn học Trí tuệ nhân tạo.

## 1. Cấu Trúc Thư Mục Dự Án Hiện Tại

```text
movie_recommendation_system/
│
├── data/
│   ├── raw/                   # Thư mục chứa dữ liệu gốc từ MovieLens 100k
│   │   ├── u.data             # Tệp chứa 100,000 bản ghi điểm đánh giá
│   │   ├── u.item             # Tệp chứa thông tin danh mục tên phim
│   │   └── README             # Tài liệu hướng dẫn gốc của bộ dữ liệu
│   │
│   └── processed/             # TỐI ƯU: Lưu trữ ma trận kề Train/Test dưới dạng nhị phân (.npy)
│                              # Giúp hệ thống nạp dữ liệu tức thì, bỏ qua bước đọc file text thô
│
├── models/                    # TỐI ƯU: Đóng gói trọng số mô hình SVD (svd_weights.pkl) sau khi học
│                              # Phục vụ chạy Demo giao diện ngay lập tức không cần huấn luyện lại
│
├── notebooks/                 # TỐI ƯU: Không gian nghiên cứu và trực quan hóa dữ liệu thô
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