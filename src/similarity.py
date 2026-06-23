import numpy as np

def compute_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Tính ma trận độ tương đồng Cosine thuần túy giữa các dòng (Users hoặc Items tùy đầu vào).
    Công thức: Sim(A, B) = (A . B) / (||A|| * ||B||)
    """
    epsilon = 1e-9  # Tránh lỗi chia cho 0
    
    # Tính tích vô hướng giữa tất cả các cặp vector dòng
    dot_product = matrix.dot(matrix.T)
    
    # Tính độ dài (L2 norm) của từng vector dòng
    norms = np.linalg.norm(matrix, axis=1)
    
    # Nhân chéo độ dài các cặp vector để làm mẫu số
    norm_matrix = norms[:, np.newaxis].dot(norms[np.newaxis, :])
    
    return dot_product / (norm_matrix + epsilon)

def compute_pearson_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Tính ma trận hệ số tương quan Pearson giữa các dòng (Users).
    Chuẩn hóa bằng cách trừ đi điểm trung bình của từng user (chỉ tính các ô đã đánh giá > 0).
    """
    epsilon = 1e-9
    mask = matrix > 0  # Đánh dấu các vị trí có dữ liệu đánh giá
    
    user_counts = mask.sum(axis=1)
    user_means = np.zeros(matrix.shape[0])
    valid_users = user_counts > 0
    user_means[valid_users] = matrix.sum(axis=1)[valid_users] / user_counts[valid_users]
    
    mean_matrix = user_means[:, np.newaxis].repeat(matrix.shape[1], axis=1)
    mean_centered = np.zeros_like(matrix)
    mean_centered[mask] = matrix[mask] - mean_matrix[mask]
    
    dot_product = mean_centered.dot(mean_centered.T)
    
    # Tính norm chỉ trên các item cùng được đánh giá (co-rated items)
    sq_matrix = mean_centered ** 2
    norm_u_sq = sq_matrix.dot(mask.T)
    norm_v_sq = mask.dot(sq_matrix.T)
    norm_matrix = np.sqrt(norm_u_sq * norm_v_sq)
    
    raw_sim = dot_product / (norm_matrix + epsilon)
    
    return raw_sim

def compute_adjusted_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Adjusted Cosine Similarity giữa các cột (Items).
    Đo độ tương đồng giữa item i và item j, nhưng trừ đi điểm trung bình của user đã xem cả hai.
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
    
    return raw_sim