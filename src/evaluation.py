import numpy as np
from typing import Any

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