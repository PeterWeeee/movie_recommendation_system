import numpy as np
from typing import Any, Tuple
import time

def compute_mae(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán sai số tuyệt đối trung bình (Mean Absolute Error - MAE).
    Đo lường độ lệch trung bình giữa điểm dự đoán và điểm thực tế.
    Tối ưu hóa tốc độ bằng cách gọi predict_batch.
    """
    users, items = np.where(test_matrix > 0)
    
    if len(users) == 0:
        return 0.0
        
    actual_ratings = test_matrix[users, items]
    predicted_ratings = np.zeros(len(users))
    
    unique_users = np.unique(users)
    for user in unique_users:
        user_mask = users == user
        items_for_user = items[user_mask]
        preds = model.predict_batch(user, items_for_user)
        predicted_ratings[user_mask] = preds
    
    return float(np.mean(np.abs(actual_ratings - predicted_ratings)))

def compute_rmse(test_matrix: np.ndarray, model: Any) -> float:
    """
    Tính toán căn bậc hai của sai số bình phương trung bình (Root Mean Squared Error - RMSE).
    Độ đo này phạt nặng hơn đối với các lỗi dự đoán lệch quá xa so với thực tế.
    Tối ưu hóa tốc độ bằng cách gọi predict_batch.
    """
    users, items = np.where(test_matrix > 0)
    
    if len(users) == 0:
        return 0.0
        
    actual_ratings = test_matrix[users, items]
    predicted_ratings = np.zeros(len(users))
    
    unique_users = np.unique(users)
    for user in unique_users:
        user_mask = users == user
        items_for_user = items[user_mask]
        preds = model.predict_batch(user, items_for_user)
        predicted_ratings[user_mask] = preds
    
    return float(np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2)))

def compute_precision_recall_at_k(train_matrix: np.ndarray, test_matrix: np.ndarray, model: Any, k: int = 10, threshold: float = 3.5) -> Tuple[float, float]:
    """
    Tính toán Precision@K và Recall@K.
    Chỉ lấy một tập mẫu user ngẫu nhiên để tăng tốc độ nếu ma trận lớn.
    """
    unique_users = np.unique(np.where(test_matrix > 0)[0])
    # Giới hạn số lượng user để tính toán nhanh (lấy tối đa 100 user ngẫu nhiên)
    rng = np.random.default_rng(42)
    if len(unique_users) > 100:
        sample_users = rng.choice(unique_users, 100, replace=False)
    else:
        sample_users = unique_users
        
    precisions = []
    recalls = []
    
    for user in sample_users:
        # Các phim thực sự được user đánh giá cao (>= threshold)
        test_items_positive = np.where(test_matrix[user] >= threshold)[0]
        
        if len(test_items_positive) == 0:
            continue
            
        # Dự đoán điểm cho TẤT CẢ các phim chưa xem trong train set (bao gồm cả test_items)
        unviewed_in_train = np.where(train_matrix[user] == 0)[0]
        if len(unviewed_in_train) == 0:
            continue
            
        preds = model.predict_batch(user, unviewed_in_train)
        
        # Lấy top K phim có điểm dự đoán cao nhất
        top_k_indices = unviewed_in_train[np.argsort(preds)[-k:][::-1]]
        
        # Số phim dự đoán nằm trong top K thực sự là phim tốt
        hits = len(set(top_k_indices) & set(test_items_positive))
        
        precisions.append(hits / k)
        recalls.append(hits / len(test_items_positive))
        
    if not precisions:
        return 0.0, 0.0
        
    return float(np.mean(precisions)), float(np.mean(recalls))

def compute_f1_at_k(precision: float, recall: float) -> float:
    """
    Tính F1-Score@K từ Precision@K và Recall@K đã tính sẵn.
    F1 là trung bình điều hòa của Precision và Recall.
    Nhận kết quả từ compute_precision_recall_at_k để tránh tính lại predict_batch.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_ndcg_at_k(train_matrix: np.ndarray, test_matrix: np.ndarray, model: Any, k: int = 10) -> float:
    """Tính toán NDCG@K."""
    unique_users = np.unique(np.where(test_matrix > 0)[0])
    rng = np.random.default_rng(42)
    sample_users = rng.choice(unique_users, min(len(unique_users), 100), replace=False)
    ndcg_scores = []
    
    for user in sample_users:
        test_items = np.where(test_matrix[user] > 0)[0]
        if len(test_items) == 0:
            continue
            
        # Dự đoán cho toàn bộ phim chưa xem trong train
        unviewed_in_train = np.where(train_matrix[user] == 0)[0]
        if len(unviewed_in_train) == 0:
            continue
            
        preds = model.predict_batch(user, unviewed_in_train)
        top_k_indices = unviewed_in_train[np.argsort(preds)[-k:][::-1]]
        
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            # Nếu phim gợi ý nằm trong test set thì lấy điểm thật, nếu không coi như 0 điểm (không quan trọng)
            true_rating = test_matrix[user, idx] if test_matrix[user, idx] > 0 else 0
            if true_rating > 0:
                dcg += (2**true_rating - 1) / np.log2(i + 2)
        
        sorted_test = np.sort(test_matrix[user, test_items])[::-1]
        idcg = 0.0
        for i in range(min(k, len(sorted_test))):
            idcg += (2**sorted_test[i] - 1) / np.log2(i + 2)
            
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
            
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

def compute_prediction_time(test_matrix: np.ndarray, model: Any) -> float:
    """
    Đo thời gian dự đoán (giây) trên tập test_matrix (chia trung bình cho mỗi user).
    Giúp so sánh tốc độ giữa các mô hình.
    """
    unique_users = np.unique(np.where(test_matrix > 0)[0])
    rng = np.random.default_rng(42)
    if len(unique_users) > 50:
        sample_users = rng.choice(unique_users, 50, replace=False)
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

def compute_training_time(train_matrix: np.ndarray, model: Any, test_matrix: np.ndarray | None = None) -> float:
    """
    Đo thời gian huấn luyện (giây) của một mô hình.
    Đối với SVD, nó đo thời gian chạy thuật toán tối ưu (fit).
    Đối với CF, nó đo thời gian tính toán ma trận tương đồng (similarity) và khởi tạo mô hình (means/baseline).
    """
    import src.recommender
    from src.similarity import compute_pearson_similarity, compute_cosine_similarity, compute_adjusted_cosine_similarity
    
    start_time = time.time()
    
    if isinstance(model, src.recommender.MatrixFactorizationSVD):
        temp_model = src.recommender.MatrixFactorizationSVD(
            num_factors=model.num_factors, lr=model.lr, reg=model.reg, epochs=model.epochs
        )
        temp_model.fit(train_matrix, test_matrix)
        
    elif isinstance(model, src.recommender.UserBasedCollaborativeFiltering):
        temp_model = src.recommender.UserBasedCollaborativeFiltering(
            k_neighbors=model.k_neighbors, prediction_mode=model.prediction_mode
        )
        if model.prediction_mode == 'basic':
            sim = compute_cosine_similarity(train_matrix)
        else:
            sim = compute_pearson_similarity(train_matrix)
        temp_model.fit(train_matrix, sim)
        
    elif isinstance(model, src.recommender.ItemBasedCollaborativeFiltering):
        temp_model = src.recommender.ItemBasedCollaborativeFiltering(
            k_neighbors=model.k_neighbors
        )
        # Kiểm tra xem mô hình hiện tại đang dùng cosine hay adjusted cosine (dựa trên tên hoặc logic)
        # Để đơn giản, ta tính luôn Adjusted Cosine vì ItemBased thường dùng Adjusted Cosine.
        sim = compute_adjusted_cosine_similarity(train_matrix)
        temp_model.fit(train_matrix, sim)
        
    return time.time() - start_time