import numpy as np
from src.data_loader import train_test_split_matrix

def get_toy_matrix() -> np.ndarray:
    """
    Tạo ma trận 10x10 bao gồm các trường hợp:
    - User 0, 1, 2 có sở thích tương đồng (thích phim 0, 1, 2).
    - User 3 có sở thích dị (thích phim 3, 4).
    - User 4 hoàn toàn không có tương đồng với ai (thích phim 5).
    - User 8 chưa đánh giá phim nào (toàn 0).
    - Item 9 chưa được ai đánh giá (toàn 0).
    - Các ô còn lại có rải rác vài rating.
    """
    matrix = np.zeros((10, 10), dtype=np.float64)
    
    # User 0, 1, 2 tương đồng nhau (thích Item 0, 1, 2)
    matrix[0, 0:3] = [5, 4, 4]
    matrix[1, 0:3] = [4, 5, 4]
    matrix[2, 0:3] = [4, 4, 5]
    matrix[0, 6] = 3
    matrix[1, 7] = 2
    
    # User 3 thích Item 3, 4
    matrix[3, 3:5] = [5, 4]
    matrix[3, 8] = 2
    
    # User 4 thích Item 5 (rất dị, không ai khác xem item 5)
    matrix[4, 5] = 5
    
    # User 5, 6, 7 đánh giá ngẫu nhiên rải rác
    matrix[5, 1] = 2
    matrix[5, 4] = 3
    matrix[5, 6] = 4
    
    matrix[6, 2] = 1
    matrix[6, 7] = 5
    matrix[6, 8] = 4
    
    matrix[7, 0] = 3
    matrix[7, 3] = 4
    matrix[7, 6] = 5
    
    # User 8 toàn 0 (mới)
    # Item 9 toàn 0 (mới)
    
    # User 9 thêm một vài rating để ma trận bớt thưa
    matrix[9, 1] = 4
    matrix[9, 5] = 1
    matrix[9, 8] = 3
    
    return matrix

def get_toy_movie_titles() -> dict:
    return {
        1: "The Matrix (Hành Động)",
        2: "Inception (Hành Động)",
        3: "Interstellar (Khoa Học Viễn Tưởng)",
        4: "Titanic (Tình Cảm)",
        5: "The Notebook (Tình Cảm)",
        6: "Obscure Indie Movie (Nghệ Thuật)",
        7: "Toy Story (Hoạt Hình)",
        8: "Finding Nemo (Hoạt Hình)",
        9: "The Conjuring (Kinh Dị)",
        10: "Unreleased Movie (Chưa Phát Hành)"
    }

def get_toy_train_test_split(test_ratio=0.2, random_seed=42):
    """
    Lấy ma trận toy và chia thành train/test.
    """
    matrix = get_toy_matrix()
    train_matrix, test_matrix = train_test_split_matrix(matrix, test_ratio=test_ratio, random_seed=random_seed)
    return train_matrix, test_matrix
