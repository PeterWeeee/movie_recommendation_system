import numpy as np

def compute_cosine_similarity(matrix):
    """
    Tính ma trận độ tương đồng Cosine giữa các dòng (Users).
    Công thức: Sim(A, B) = (A . B) / (||A|| * ||B||)
    """
    epsilon = 1e-9  # Tránh lỗi chia cho 0, công cái này vào mẫu số công thức Sim(A,B)
    
    # Tính tích vô hướng giữa tất cả các cặp vector dòng: (num_users x num_users)
    dot_product = matrix.dot(matrix.T)
    
    # Tính độ dài (L2 norm) của từng vector dòng: (num_users,)
    norms = np.linalg.norm(matrix, axis=1)
    
    # Nhân chéo độ dài các cặp vector để làm mẫu số: (num_users x num_users)
    norm_matrix = norms[:, np.newaxis].dot(norms[np.newaxis, :])
    
    # Trả về ma trận độ tương đồng
    return dot_product / (norm_matrix + epsilon)

def compute_pearson_similarity(matrix):
    """
    Tính ma trận hệ số tương quan Pearson giữa các dòng (Users).
    Chuẩn hóa bằng cách trừ đi điểm trung bình của từng user (chỉ tính các ô đã đánh giá > 0).
    """
    epsilon = 1e-9
    mask = matrix > 0  # Đánh dấu các vị trí có dữ liệu đánh giá, ô nào có điểm đánh giá > 0 là true
    
    # Đếm số lượng phim từng user đã xem
    user_counts = mask.sum(axis=1)
    
    # Tính điểm đánh giá trung bình của từng user
    user_means = np.zeros(matrix.shape[0])
    valid_users = user_counts > 0
    user_means[valid_users] = matrix.sum(axis=1)[valid_users] / user_counts[valid_users]
    
    # Trừ điểm đánh giá cho trung bình (Mean Centering)
    # Tạo ma trận trung bình cùng kích thước với ma trận gốc
    mean_matrix = user_means[:, np.newaxis].repeat(matrix.shape[1], axis=1)
    mean_centered = np.zeros_like(matrix)
    
    # Chỉ trừ tại những vị trí user thực sự đã cho điểm
    mean_centered[mask] = matrix[mask] - mean_matrix[mask]
    
    # Áp dụng công thức Cosine trên ma trận đã chuẩn hóa tâm để ra hệ số Pearson
    dot_product = mean_centered.dot(mean_centered.T)
    norms = np.linalg.norm(mean_centered, axis=1)
    norm_matrix = norms[:, np.newaxis].dot(norms[np.newaxis, :])
    
    return dot_product / (norm_matrix + epsilon)