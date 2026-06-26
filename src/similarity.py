import numpy as np

def compute_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Tính ma trận độ tương đồng Cosine giữa các dòng (Users hoặc Items).
    Sử dụng công thức gốc: norm được tính trên toàn bộ vector (global norm).
    """
    epsilon = 1e-9  # Tránh lỗi chia cho 0
    
    # Tích vô hướng (dot product) tự nhiên chỉ tính trên các ô cùng có giá trị > 0
    dot_product = matrix.dot(matrix.T)
    
    # Tính norm trên toàn bộ chiều của vector (công thức Cosine nguyên thủy)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_matrix = norm.dot(norm.T)
    
    return dot_product / (norm_matrix + epsilon)

def compute_pearson_similarity(matrix: np.ndarray, gamma: int = 50) -> np.ndarray:
    """
    Tính ma trận hệ số tương quan Pearson giữa các dòng (Users).
    Sử dụng significance weighting để giảm trọng số của những cặp có quá ít co-rated items.
    """
    epsilon = 1e-9
    mask = matrix > 0  # Đánh dấu các vị trí có dữ liệu đánh giá
    #tính trung bình đánh giá của từng user
    user_counts = mask.sum(axis=1)
    user_means = np.zeros(matrix.shape[0])
    valid_users = user_counts > 0
    user_means[valid_users] = matrix.sum(axis=1)[valid_users] / user_counts[valid_users]
    #tạo ma trận trung bình đánh giá của từng user
    mean_matrix = user_means[:, np.newaxis].repeat(matrix.shape[1], axis=1)
    mean_centered = np.zeros_like(matrix)
    mean_centered[mask] = matrix[mask] - mean_matrix[mask]
    # tính tích vô hướng của các vector đã trừ trung bình
    dot_product = mean_centered.dot(mean_centered.T)
    
    # Tính norm chỉ trên các item cùng được đánh giá (co-rated items)
    sq_matrix = mean_centered ** 2
    norm_u_sq = sq_matrix.dot(mask.T)
    norm_v_sq = mask.dot(sq_matrix.T)
    norm_matrix = np.sqrt(norm_u_sq * norm_v_sq)
    
    raw_sim = dot_product / (norm_matrix + epsilon)
    
    # Áp dụng Significance Weighting
    mask_int = mask.astype(int)
    co_rated_counts = mask_int.dot(mask_int.T)
    significance_weights = np.minimum(co_rated_counts, gamma) / gamma
    
    return raw_sim * significance_weights

def compute_adjusted_cosine_similarity(matrix: np.ndarray, gamma: int = 50) -> np.ndarray:
    """
    Tính ma trận Adjusted Cosine Similarity giữa các cột (Items).
    Sử dụng significance weighting.
    Đầu vào: matrix (num_users x num_items)
    Đầu ra: ma trận tương đồng (num_items x num_items)
    """
    epsilon = 1e-9
    mask = matrix > 0
    
    user_counts = mask.sum(axis=1)
    user_means = np.zeros(matrix.shape[0])
    valid_users = user_counts > 0
    user_means[valid_users] = matrix.sum(axis=1)[valid_users] / user_counts[valid_users]
    
    mean_matrix = user_means[:, np.newaxis].repeat(matrix.shape[1], axis=1)
    mean_centered = np.zeros_like(matrix)
    mean_centered[mask] = matrix[mask] - mean_matrix[mask]
    
    # Tính Cosine Similarity giữa các cột của mean_centered
    # mean_centered.T có kích thước (num_items x num_users)
    item_matrix = mean_centered.T
    
    dot_product = item_matrix.dot(item_matrix.T)
    
    # Tính norm chỉ trên các user cùng đánh giá cả 2 item (co-rated users)
    mask_item = mask.T
    sq_matrix_item = item_matrix ** 2
    norm_i_sq = sq_matrix_item.dot(mask_item.T)
    norm_j_sq = mask_item.dot(sq_matrix_item.T)
    norm_matrix = np.sqrt(norm_i_sq * norm_j_sq)
    
    raw_sim = dot_product / (norm_matrix + epsilon)
    
    # Áp dụng Significance Weighting
    mask_item_int = mask_item.astype(int)
    co_rated_counts = mask_item_int.dot(mask_item_int.T)
    significance_weights = np.minimum(co_rated_counts, gamma) / gamma
    
    return raw_sim * significance_weights