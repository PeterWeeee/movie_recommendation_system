import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging

from src.data_loader import load_movie_titles, load_matrix
from src.evaluation import compute_mae, compute_rmse

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    # Xác định đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
    item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path) or not os.path.exists(item_path):
        logging.error("[LỖI] Không tìm thấy dữ liệu MovieLens 100k!")
        return

    logging.info("--- BẮT ĐẦU HỆ THỐNG GỢI Ý LAI (TERMINAL) ---")
    
    # 1. Tải thông tin tên phim
    movie_titles = load_movie_titles(item_path)
    logging.info(f"[Bước 1/3] Đã tải thành công danh mục gồm {len(movie_titles)} bộ phim.")

    # 2. Tải các file cấu hình và models đã được huấn luyện
    logging.info("[Bước 2/3] Đang nạp ma trận và các mô hình từ thư mục models/...")
    train_matrix_path = os.path.join(processed_dir, 'train_matrix.npy')
    test_matrix_path = os.path.join(processed_dir, 'test_matrix.npy')
    
    if not os.path.exists(train_matrix_path) or not os.path.exists(os.path.join(models_dir, 'user_cf.pkl')):
        logging.error("[LỖI] Chưa tìm thấy mô hình huấn luyện!")
        logging.error("Vui lòng chạy lệnh: python scripts/train_pipeline.py")
        return
        
    train_matrix = load_matrix(train_matrix_path)
    test_matrix = load_matrix(test_matrix_path)
    
    with open(os.path.join(models_dir, 'user_cf.pkl'), 'rb') as f:
        user_cf = pickle.load(f)
        
    with open(os.path.join(models_dir, 'item_cf.pkl'), 'rb') as f:
        item_cf = pickle.load(f)
        
    from src.recommender import MatrixFactorizationSVD
    svd_model = MatrixFactorizationSVD()
    svd_model.load_model(os.path.join(models_dir, 'svd_weights.pkl'))
    
    logging.info("-> Khởi tạo thành công các mô hình: User-Based CF, Item-Based CF, SVD.")

    # 3. Kịch bản chạy thử nghiệm gợi ý phim cá nhân hóa cho User cố định
    logging.info("[Bước 3/3] --- ĐANG MÔ PHỎNG GỢI Ý CHO NGƯỜI DÙNG CỤ THỂ ---")
    target_user_id = 1
    target_user_idx = target_user_id - 1
    logging.info(f"Đang tìm danh sách phim gợi ý tối ưu nhất cho User_ID: {target_user_id} bằng thuật toán SVD...")
    
    # Lọc ra các vị trí phim chưa xem
    unviewed_items = np.where(train_matrix[target_user_idx] == 0)[0]
    
    if len(unviewed_items) == 0:
        logging.info("Người dùng này đã xem tất cả các phim.")
    else:
        # Sử dụng Vectorization để dự đoán
        preds = svd_model.predict_batch(target_user_idx, unviewed_items)
        
        # Sắp xếp và chọn Top 5
        top_n_indices = np.argsort(preds)[-5:][::-1]
        top_item_indices = unviewed_items[top_n_indices]
        top_scores = preds[top_n_indices]
        
        logging.info(f"Top 5 bộ phim được gợi ý nhiều nhất:")
        for rank, (item_idx, score) in enumerate(zip(top_item_indices, top_scores), 1):
            movie_name = movie_titles.get(int(item_idx) + 1, f"Phim ID {int(item_idx) + 1}")
            logging.info(f" Hạng {rank}: {movie_name} -> Điểm dự kiến: {score:.2f} sao")
            
    logging.info("--- HOÀN THÀNH CHƯƠNG TRÌNH THỬ NGHIỆM ---")

if __name__ == '__main__':
    main()