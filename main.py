import os
import numpy as np
import pandas as pd
from src.data_loader import load_raw_data, load_movie_titles, get_or_create_processed_matrices
from src.similarity import compute_pearson_similarity
from src.recommender import UserBasedCollaborativeFiltering, MatrixFactorizationSVD
from src.evaluation import compute_mae, compute_rmse

def main():
    # 1. Xác định đường dẫn tuyệt đối dựa theo vị trí file main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
    item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    model_path = os.path.join(base_dir, 'models', 'svd_weights.pkl')
    
    if not os.path.exists(data_path) or not os.path.exists(item_path):
        print("[LỖI] Không tìm thấy dữ liệu MovieLens 100k!")
        print(f"Vui lòng kiểm tra lại file tại thư mục data/raw/")
        return

    print("--- BẮT ĐẦU HỆ THỐNG GỢI Ý MOVIELENS 100K (TERMINAL) ---")
    
    # 2. Tải thông tin tên phim từ u.item
    movie_titles = load_movie_titles(item_path)
    print(f"\n[Bước 1/4] Đã tải thành công danh mục gồm {len(movie_titles)} bộ phim.")

    # 3. Tải ma trận dữ liệu tối ưu (Tận dụng thư mục data/processed/)
    print("\n[Bước 2/4] Đang nạp ma trận dữ liệu từ thư mục data/processed/...")
    # Hàm này tự động tạo file .npy nếu chạy lần đầu, hoặc nạp trực tiếp nếu đã có file
    train_matrix, test_matrix = get_or_create_processed_matrices(data_path, processed_dir, test_ratio=0.2)
    print(f"-> Ma trận Huấn luyện (Train set): {train_matrix.shape[0]} users x {train_matrix.shape[1]} items.")
    
    # Tính toán độ thưa thớt (Sparsity) để in ra Terminal phục vụ báo cáo số liệu
    total_elements = train_matrix.shape[0] * train_matrix.shape[1]
    nonzero_elements = np.count_nonzero(train_matrix)
    sparsity = (1 - nonzero_elements / total_elements) * 100
    print(f"-> Mức độ thưa thớt hiện tại của ma trận kề: {sparsity:.2f}%")

    # 4. Tính toán ma trận tương đồng Pearson cho User-Based CF
    print("\n[Bước 3/4] Đang tính toán ma trận tương đồng Pearson...")
    pearson_sim = compute_pearson_similarity(train_matrix)

    # 5. Khởi tạo và Huấn luyện Mô hình 1: User-Based CF (KNN)
    print("\n[Bước 4/4] Đang cấu hình mô hình User-Based CF (K=40)...")
    user_cf = UserBasedCollaborativeFiltering(k_neighbors=40)
    user_cf.fit(train_matrix, pearson_sim)
    
    print("-> Đang kiểm thử hiệu năng User-Based CF...")
    user_cf_mae = compute_mae(test_matrix, user_cf)
    user_cf_rmse = compute_rmse(test_matrix, user_cf)

    # 6. Khởi tạo và Huấn luyện/Tải Mô hình 2: Matrix Factorization SVD (Tận dụng thư mục models/)
    print("\n[Bước 4/4 Tiếp theo] Đang thiết lập mô hình Matrix Factorization (SVD)...")
    svd_model = MatrixFactorizationSVD(num_factors=20, lr=0.005, reg=0.02, epochs=20)
    
    # Kiểm tra xem mô hình SVD đã từng học và lưu trọng số trong thư mục models/ chưa
    if os.path.exists(model_path):
        print("-> Phát hiện file trọng số cũ. Đang nạp nhanh mô hình SVD từ thư mục models/...")
        svd_model.load_model(model_path)
    else:
        print("-> Chưa có file trọng số. Tiến hành huấn luyện mô hình SVD bằng thuật toán SGD (20 epochs)...")
        svd_model.fit(train_matrix)
        print("-> Đang đóng gói và lưu trọng số mô hình vào thư mục models/ để tái sử dụng...")
        svd_model.save_model(model_path)
    
    print("-> Đang kiểm thử hiệu năng SVD...")
    svd_mae = compute_mae(test_matrix, svd_model)
    svd_rmse = compute_rmse(test_matrix, svd_model)

    # 7. In bảng số liệu so sánh kết quả kiểm thử sai số công thức
    print("\n" + "="*50)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ ĐỀ TÀI (ĐỘ LỆCH SAI SỐ)':^50}")
    print("="*50)
    
    results_data = {
        'Thuật toán (Từ đầu)': ['User-Based CF (Pearson)', 'Matrix Factorization (SVD)'],
        'MAE (Càng thấp càng tốt)': [f"{user_cf_mae:.4f}", f"{svd_mae:.4f}"],
        'RMSE (Càng thấp càng tốt)': [f"{user_cf_rmse:.4f}", f"{svd_rmse:.4f}"]
    }
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))
    print("="*50)

    # 8. Kịch bản chạy thử nghiệm gợi ý phim cá nhân hóa cho User cố định trên Terminal
    print("\n--- ĐANG MÔ PHỎNG GỢI Ý CHO NGƯỜI DÙNG CỤ THỂ ---")
    target_user_id = 1
    target_user_idx = target_user_id - 1
    print(f"Đang tìm danh sách phim gợi ý tối ưu nhất cho User_ID: {target_user_id}...")
    
    # Lọc ra các vị trí phim chưa xem (giá trị bằng 0 trong tập huấn luyện)
    unviewed_items = np.where(train_matrix[target_user_idx] == 0)[0]
    
    # Sử dụng mô hình SVD đã tối ưu để chấm điểm dự kiến cho các phim chưa xem
    predicted_ratings = []
    for item_idx in unviewed_items:
        pred_score = svd_model.predict_rating(target_user_idx, item_idx)
        predicted_ratings.append((item_idx, pred_score))
        
    # Sắp xếp chọn ra Top 5 bộ phim có điểm dự kiến cao nhất
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    top_5_recommendations = predicted_ratings[:5]
    
    print(f"\nTop 5 bộ phim được gợi ý nhiều nhất cho User_ID {target_user_id}:")
    for rank, (item_idx, score) in enumerate(top_5_recommendations, 1):
        movie_name = movie_titles.get(item_idx + 1, f"Phim ID {item_idx + 1}")
        print(f" Hạng {rank}: {movie_name} -> Điểm dự kiến: {score:.2f} sao")
        
    print("\n--- HOÀN THÀNH CHƯƠNG TRÌNH THỬ NGHIỆM ---")

if __name__ == '__main__':
    main()