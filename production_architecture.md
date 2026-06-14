# Đề Xuất Kiến Trúc Triển Khai Thực Tế (Production Architecture)

Khi mở rộng quy mô hệ thống gợi ý từ môi trường thử nghiệm (MovieLens 100k, file `.npy` và `.pkl`) lên quy mô doanh nghiệp với hàng triệu người dùng và bộ phim, kiến trúc lưu trữ và xử lý hiện tại sẽ gặp phải hai thách thức lớn:
1. **Tràn bộ nhớ RAM**: Không thể nạp một ma trận khổng lồ hàng GB vào RAM để tính KNN hay SVD mỗi lần khởi động.
2. **Độ trễ thời gian thực**: Việc duyệt qua mảng `.npy` hay tính toán tích vô hướng (dot product) trên CPU cho hàng triệu Item sẽ gây giật lag (Latency cao), ảnh hưởng trải nghiệm người dùng.

Dưới đây là đề xuất thiết kế kiến trúc phân tầng chuyên nghiệp:

## 1. Cơ Sở Dữ Liệu Quan Hệ (RDBMS) - PostgreSQL / SQL Server
**Mục đích**: Quản lý dữ liệu có cấu trúc.
* Lưu thông tin người dùng (User Profile), đăng nhập, cài đặt.
* Lưu danh mục bộ phim (Movie Catalog) với đầy đủ metadata (Title, Year, Cast, Directors).
* Lưu lịch sử đánh giá phim (Ratings History). Khi User rate phim mới, dữ liệu sẽ được ghi thẳng vào bảng SQL.

## 2. Vector Database (Milvus / Pinecone)
**Mục đích**: Tối ưu hóa tính toán tương đồng (Similarity Search) siêu tốc.
* Khi mô hình Matrix Factorization (SVD) hoàn thành huấn luyện offline, nó sẽ sinh ra hai ma trận: **Latent Factors của User ($P_u$)** và **Latent Factors của Item ($Q_i$)**.
* Các vector đặc trưng $Q_i$ này sẽ được export và lưu trữ thẳng vào **Vector Database**.
* Khi một User $u$ cần gợi ý phim, hệ thống lấy vector $P_u$ của họ truy vấn vào Vector DB. Database sẽ sử dụng thuật toán **Approximate Nearest Neighbor (ANN)** (như HNSW hay Faiss) để tìm ra top các $Q_i$ có độ tương đồng Cosine hoặc Dot Product cao nhất với $P_u$ trong khoảng thời gian tính bằng **mili-giây** (ms).

## 3. Kiến Trúc Pipeline Gợi Ý
1. **Offline Training (Batch Processing)**:
   * Chạy định kỳ mỗi đêm bằng Apache Spark hoặc Airflow.
   * Lấy toàn bộ Ratings từ PostgreSQL -> Xây dựng tập Train -> Cập nhật mô hình SVD -> Lưu $P_u$, $Q_i$ vào Data Warehouse hoặc cập nhật Vector DB.
2. **Online Inference (Real-time Serving)**:
   * **Bước 1**: App gửi `user_id` tới Backend API (FastAPI / Node.js).
   * **Bước 2**: Backend kiểm tra Cache (Redis). Nếu có, trả về ngay.
   * **Bước 3**: Nếu không có, lấy vector $P_u$ của user đẩy vào Vector Database.
   * **Bước 4**: Lọc bỏ các ID phim đã xem (lấy nhanh từ Redis/PostgreSQL).
   * **Bước 5**: Vector DB trả về Top-N `item_id`.
   * **Bước 6**: Backend map `item_id` với PostgreSQL để lấy tên phim, ảnh bìa trả về cho App.

---
*Mô hình này giúp hệ thống chịu tải tốt, tìm kiếm cực nhanh và đảm bảo tính cá nhân hóa theo đúng lý thuyết của Lọc cộng tác SVD.*
