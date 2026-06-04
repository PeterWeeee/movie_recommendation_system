import numpy as np
import pickle
import os
from typing import Optional

class UserBasedCollaborativeFiltering:
    """
    Hệ thống gợi ý Lọc cộng tác dựa trên Người dùng (User-based Collaborative Filtering).
    Sử dụng phương pháp K-Người láng giềng gần nhất (KNN).
    """
    def __init__(self, k_neighbors: int = 40) -> None:
        self.k_neighbors = k_neighbors
        self.train_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None

    def fit(self, train_matrix: np.ndarray, similarity_matrix: np.ndarray) -> None:
        """
        Lưu trữ ma trận huấn luyện và ma trận độ tương đồng đã tính trước.
        """
        self.train_matrix = train_matrix
        self.similarity_matrix = similarity_matrix
        
        # Tính điểm trung bình của từng user (chỉ tính trên các ô > 0)
        mask = train_matrix > 0 #gán true cho những ô có điểm và false cho ô không có điểm (0)
        user_counts = mask.sum(axis=1) #đếm số lượng phim mà user đã đánh giá
        self.user_means = np.zeros(train_matrix.shape[0]) #tạo ma trận để chứa chứa điểm trung bình cho từng users
        valid_users = user_counts > 0 #mục đích để lấy ra những users đã đánh giá ít nhất một phim
        self.user_means[valid_users] = train_matrix.sum(axis=1)[valid_users] / user_counts[valid_users]

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        """
        Dự đoán điểm đánh giá của user_idx cho item_idx bằng công thức trung bình có trọng số.
        """
        assert self.train_matrix is not None and self.similarity_matrix is not None and self.user_means is not None, \
            "Model chưa được huấn luyện. Hãy gọi fit() trước."
        # Nếu bộ phim này đã có điểm trong tập Train, trả về điểm đó luôn
        if self.train_matrix[user_idx, item_idx] > 0:
            return self.train_matrix[user_idx, item_idx]
            
        # Tìm tất cả những người dùng khác đã đánh giá bộ phim này
        other_users = np.where(self.train_matrix[:, item_idx] > 0)[0]
        
        # Nếu chưa ai xem phim này thì sẽ trả về điểm trung bình user này, nếu user này chưa xem thì trả về
        #điểm trung bình an toàn là 3.0
        if len(other_users) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0
            
        # Lấy độ tương đồng giữa user hiện tại và các user khác đã xem phim
        similarities = self.similarity_matrix[user_idx, other_users]
        
        # Sắp xếp và lấy ra K người láng giềng có độ tương đồng cao nhất
        top_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        top_similarities = similarities[top_indices]
        top_other_users = other_users[top_indices]
        
        # Tính toán dự đoán chuẩn hóa theo trung bình (Adjusted Prediction)
        sim_sum = np.sum(np.abs(top_similarities))
        if sim_sum == 0:
            return self.user_means[user_idx]
            
        # Lấy độ lệch điểm của láng giềng (điểm thực tế - điểm trung bình của họ)
        ratings = self.train_matrix[top_other_users, item_idx]
        means = self.user_means[top_other_users]
        rating_diffs = ratings - means
        
        # Công thức dự đoán có trọng số
        predicted_rating = self.user_means[user_idx] + (np.sum(top_similarities * rating_diffs) / sim_sum)
        
        # Giới hạn thang điểm trong khoảng [1, 5]
        return np.clip(predicted_rating, 1.0, 5.0)


class MatrixFactorizationSVD:
    """
    Hệ thống gợi ý dựa trên Phân rã ma trận (Matrix Factorization) sử dụng thuật toán SVD.
    Tối ưu hóa các tham số thông qua thuật toán Stochastic Gradient Descent (SGD).
    """
    def __init__(self, num_factors: int = 20, lr: float = 0.005, reg: float = 0.02, epochs: int = 20) -> None:
        self.num_factors = num_factors  # Số lượng đặc trưng ẩn (K)
        self.lr = lr                    # Tốc độ học (Learning rate)
        self.reg = reg                  # Hệ số kiểm soát quá khớp (Regularization)
        self.epochs = epochs            # Số vòng lặp huấn luyện
        self.mu: float = 0.0            # Điểm trung bình toàn bộ hệ thống
        self.b_u: Optional[np.ndarray] = None                 # Độ lệch (bias) của người dùng
        self.b_i: Optional[np.ndarray] = None                 # Độ lệch (bias) của bộ phim
        self.P: Optional[np.ndarray] = None                   # Ma trận đặc trưng người dùng
        self.Q: Optional[np.ndarray] = None                   # Ma trận đặc trưng bộ phim

    def fit(self, train_matrix: np.ndarray) -> None:
        """
        Huấn luyện mô hình tìm ma trận P, Q và các hệ số bias bằng SGD.
        """
        num_users, num_items = train_matrix.shape
        nonzero_indices = np.argwhere(train_matrix > 0)
        
        # Tính điểm trung bình tổng của các ô có dữ liệu
        self.mu = train_matrix[train_matrix > 0].mean()
        
        # Khởi tạo ngẫu nhiên các ma trận đặc trưng ẩn và bias
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        self.P = np.random.normal(0, 0.1, (num_users, self.num_factors))
        self.Q = np.random.normal(0, 0.1, (num_items, self.num_factors))
        
        # Vòng lặp tối ưu hóa qua từng Epoch
        for epoch in range(self.epochs):
            # Xáo trộn ngẫu nhiên dữ liệu để thuật toán SGD đạt hiệu quả cao nhất
            np.random.shuffle(nonzero_indices)
            
            for u, i in nonzero_indices:
                r_ui = train_matrix[u, i]
                
                # Dự đoán điểm hiện tại bằng tích vô hướng và bias: r^ = mu + b_u + b_i + P_u . Q_i
                pred_r_ui = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                
                # Tính toán sai số thực tế
                err = r_ui - pred_r_ui
                
                # Cập nhật các trọng số bias theo hướng giảm thiểu sai số
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                # Cập nhật hai ma trận đặc trưng ẩn P và Q
                # Tối ưu hóa: Loại bỏ việc dùng .copy() để giảm tiêu hao tài nguyên bộ nhớ
                p_u = self.P[u]
                q_i = self.Q[i]
                
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        """
        Dự đoán điểm số dựa trên các tham số ma trận đặc trưng đã học được.
        """
        assert self.b_u is not None and self.b_i is not None and self.P is not None and self.Q is not None, \
            "Model chưa được huấn luyện. Hãy gọi fit() trước."
        pred_r_ui = self.mu + self.b_u[user_idx] + self.b_i[item_idx] + np.dot(self.P[user_idx], self.Q[item_idx])
        return float(np.clip(pred_r_ui, 1.0, 5.0))
    
    def save_model(self, model_path: str) -> None:
        """Lưu toàn bộ trọng số huấn luyện của SVD vào file định dạng pkl"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'mu': self.mu, 'b_u': self.b_u, 'b_i': self.b_i, 
                'P': self.P, 'Q': self.Q, 'num_factors': self.num_factors
            }, f)

    def load_model(self, model_path: str) -> bool:
        """Tải các trọng số đã lưu lên bộ nhớ máy tính cực nhanh"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.mu = data['mu']
                self.b_u = data['b_u']
                self.b_i = data['b_i']
                self.P = data['P']
                self.Q = data['Q']
                self.num_factors = data['num_factors']
            return True
        return False