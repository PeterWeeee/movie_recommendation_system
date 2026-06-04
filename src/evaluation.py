import numpy as np
from typing import Any

def compute_mae(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán sai số tuyệt đối trung bình (Mean Absolute Error - MAE).
    Đo lường độ lệch trung bình giữa điểm dự đoán và điểm thực tế.
    """
    # Tìm tọa độ của tất cả các ô có dữ liệu trong tập Test
    nonzero_positions = np.argwhere(test_matrix > 0)
    total_samples = len(nonzero_positions)
    
    if total_samples == 0:
        return 0.0
        
    absolute_errors = []
    
    # Duyệt qua từng cặp (User, Item) trong tập dữ liệu kiểm thử
    for u, i in nonzero_positions:
        actual_rating = test_matrix[u, i]
        # Gọi hàm dự đoán của mô hình (UserBasedCF hoặc SVD)
        predicted_rating = model.predict_rating(u, i)
        
        # Tính trị tuyệt đối của sai số
        error = abs(actual_rating - predicted_rating)
        absolute_errors.append(error)
        
    # Trả về trung bình cộng của sai số tuyệt đối
    return float(np.mean(absolute_errors))

def compute_rmse(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán căn bậc hai của sai số bình phương trung bình (Root Mean Squared Error - RMSE).
    Độ đo này phạt nặng hơn đối với các lỗi dự đoán lệch quá xa so với thực tế.
    """
    nonzero_positions = np.argwhere(test_matrix > 0)
    total_samples = len(nonzero_positions)
    
    if total_samples == 0:
        return 0.0
        
    squared_errors = []
    
    for u, i in nonzero_positions:
        actual_rating = test_matrix[u, i]
        predicted_rating = model.predict_rating(u, i)
        
        # Tính bình phương của sai số
        error_sq = (actual_rating - predicted_rating) ** 2
        squared_errors.append(error_sq)
        
    # Tính trung bình cộng các bình phương rồi lấy căn bậc hai
    return float(np.sqrt(np.mean(squared_errors)))