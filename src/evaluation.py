import numpy as np
from typing import Any, Tuple
import time

def compute_mae(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán sai số tuyệt đối trung bình (Mean Absolute Error - MAE).
    Đo lường độ lệch trung bình giữa điểm dự đoán và điểm thực tế.
    Tối ưu hóa tốc độ bằng cách gọi predict_batch.
    """
    nonzero_positions = np.argwhere(test_matrix > 0)
    
    if len(nonzero_positions) == 0:
        return 0.0
        
    actual_ratings = test_matrix[nonzero_positions[:, 0], nonzero_positions[:, 1]]
    predicted_ratings = np.zeros(len(nonzero_positions))
    
    unique_users = np.unique(nonzero_positions[:, 0])
    for user in unique_users:
        user_mask = nonzero_positions[:, 0] == user
        items_for_user = nonzero_positions[user_mask, 1]
        preds = model.predict_batch(user, items_for_user)
        predicted_ratings[user_mask] = preds
    
    return float(np.mean(np.abs(actual_ratings - predicted_ratings)))

def compute_rmse(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán căn bậc hai của sai số bình phương trung bình (Root Mean Squared Error - RMSE).
    Độ đo này phạt nặng hơn đối với các lỗi dự đoán lệch quá xa so với thực tế.
    Tối ưu hóa tốc độ bằng cách gọi predict_batch.
    """
    nonzero_positions = np.argwhere(test_matrix > 0)
    
    if len(nonzero_positions) == 0:
        return 0.0
        
    actual_ratings = test_matrix[nonzero_positions[:, 0], nonzero_positions[:, 1]]
    predicted_ratings = np.zeros(len(nonzero_positions))
    
    unique_users = np.unique(nonzero_positions[:, 0])
    for user in unique_users:
        user_mask = nonzero_positions[:, 0] == user
        items_for_user = nonzero_positions[user_mask, 1]
        preds = model.predict_batch(user, items_for_user)
        predicted_ratings[user_mask] = preds
    
    return float(np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2)))

def compute_precision_recall_at_k(train_matrix: np.ndarray, test_matrix: np.ndarray, model: Any, k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
    """
    Tính toán Precision@K và Recall@K.
    Chỉ lấy một tập mẫu user ngẫu nhiên để tăng tốc độ nếu ma trận lớn.
    """
    unique_users = np.unique(np.argwhere(test_matrix > 0)[:, 0])
    # Giới hạn số lượng user để tính toán nhanh (lấy tối đa 100 user ngẫu nhiên)
    np.random.seed(42)
    if len(unique_users) > 100:
        sample_users = np.random.choice(unique_users, 100, replace=False)
    else:
        sample_users = unique_users
        
    precisions = []
    recalls = []
    
    for user in sample_users:
        # Tìm các phim người dùng đã xem trong test set và có điểm số cao (>= threshold)
        test_items = np.where(test_matrix[user] >= threshold)[0]
        if len(test_items) == 0:
            continue
            
        # Tìm các phim chưa xem trong train set
        unviewed_items = np.where(train_matrix[user] == 0)[0]
        if len(unviewed_items) == 0:
            continue
            
        # Dự đoán điểm cho các phim chưa xem
        preds = model.predict_batch(user, unviewed_items)
        
        # Lấy top K phim có điểm dự đoán cao nhất
        top_k_indices = unviewed_items[np.argsort(preds)[-k:][::-1]]
        
        # Tính số lượng phim gợi ý thực sự có trong test set và đạt yêu cầu
        hits = len(set(top_k_indices) & set(test_items))
        
        precisions.append(hits / k)
        recalls.append(hits / len(test_items))
        
    if not precisions:
        return 0.0, 0.0
        
    return float(np.mean(precisions)), float(np.mean(recalls))

def compute_prediction_time(test_matrix: np.ndarray, model: Any) -> float:
    """
    Đo thời gian dự đoán (giây) trên tập test_matrix (chia trung bình cho mỗi user).
    Giúp so sánh tốc độ giữa các mô hình.
    """
    unique_users = np.unique(np.argwhere(test_matrix > 0)[:, 0])
    np.random.seed(42)
    if len(unique_users) > 50:
        sample_users = np.random.choice(unique_users, 50, replace=False)
    else:
        sample_users = unique_users
        
    start_time = time.time()
    
    for user in sample_users:
        items = np.where(test_matrix[user] > 0)[0]
        if len(items) > 0:
            model.predict_batch(user, items)
            
    end_time = time.time()
    total_time = end_time - start_time
    return total_time / len(sample_users) if len(sample_users) > 0 else 0.0