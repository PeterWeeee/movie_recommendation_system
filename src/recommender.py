import numpy as np
import pickle
import os
from typing import Optional, Dict, List, Tuple

class BiasedPredictor:
    """
    Mô hình Thống kê nền (Biased Predictor).
    Dự đoán điểm số dựa trên xu hướng toàn cục: r^ = mu + b_u + b_i
    """
    def __init__(self) -> None:
        self.mu: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None

    def fit(self, train_matrix: np.ndarray) -> None:
        num_users, num_items = train_matrix.shape
        nonzero_mask = train_matrix > 0

        if not np.any(nonzero_mask):
            self.mu = 3.0
            self.b_u = np.zeros(num_users)
            self.b_i = np.zeros(num_items)
            return

        self.mu = train_matrix[nonzero_mask].mean()

        # --- Vectorized b_i (thay thế vòng lặp for item) ---
        # Tính tổng (r_ui - mu) cho mỗi item và chia cho số lượng ratings != 0
        item_counts = nonzero_mask.sum(axis=0)  # shape: (num_items,)
        item_sum = (train_matrix - self.mu) * nonzero_mask  # chỉ giữ vị trí có rating
        # Tránh chia cho 0: item không có rating giữ b_i = 0
        self.b_i = np.where(item_counts > 0, item_sum.sum(axis=0) / np.maximum(item_counts, 1), 0.0)

        # --- Vectorized b_u (thay thế vòng lặp for user) ---
        # Tính tổng (r_ui - mu - b_i) cho mỗi user và chia cho số lượng ratings != 0
        user_counts = nonzero_mask.sum(axis=1)  # shape: (num_users,)
        # broadcast b_i theo hàng: shape (num_users, num_items)
        user_sum = (train_matrix - self.mu - self.b_i[np.newaxis, :]) * nonzero_mask
        self.b_u = np.where(user_counts > 0, user_sum.sum(axis=1) / np.maximum(user_counts, 1), 0.0)

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        assert self.b_u is not None and self.b_i is not None, "Mô hình chưa được huấn luyện!"
        pred = self.mu + self.b_u[user_idx] + self.b_i[item_idx]
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        assert self.b_u is not None and self.b_i is not None, "Mô hình chưa được huấn luyện!"
        preds = self.mu + self.b_u[user_idx] + self.b_i[item_indices]
        return np.clip(preds, 1.0, 5.0)


class UserBasedCollaborativeFiltering:
    """
    Hệ thống gợi ý Lọc cộng tác dựa trên Người dùng (User-based Collaborative Filtering).
    Hỗ trợ 2 loại similarity: Cosine và Pearson.
    Mặc định prediction dùng trung bình có trọng số (weighted average) thuần túy.
    """
    def __init__(self, k_neighbors: int = 40, prediction_mode: str = 'basic') -> None:
        """
        prediction_mode: 'basic' (weighted average thuần), 'means' (KNN with Means),
                         'biased_baseline' (KNN with Biased Baseline) - giữ để tương thích ngược
        """
        self.k_neighbors = k_neighbors
        self.prediction_mode = prediction_mode
        self.train_matrix: Optional[np.ndarray] = None
        # Lưu 2 similarity matrix riêng biệt
        self.cosine_similarity_matrix: Optional[np.ndarray] = None
        self.pearson_similarity_matrix: Optional[np.ndarray] = None
        # Matrix đang được dùng
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None
        self.baseline_predictor: Optional[BiasedPredictor] = None

    def _compute_user_means(self, train_matrix: np.ndarray) -> None:
        mask = train_matrix > 0
        user_counts = mask.sum(axis=1)
        self.user_means = np.zeros(train_matrix.shape[0])
        valid_users = user_counts > 0
        self.user_means[valid_users] = train_matrix.sum(axis=1)[valid_users] / user_counts[valid_users]

    def fit(self, train_matrix: np.ndarray, similarity_matrix: np.ndarray) -> None:
        """Fit với Pearson similarity (mặc định, tương thích ngược)."""
        self.train_matrix = train_matrix
        self.pearson_similarity_matrix = similarity_matrix
        self.similarity_matrix = similarity_matrix
        self._compute_user_means(train_matrix)
        if self.prediction_mode == 'biased_baseline':
            self.baseline_predictor = BiasedPredictor()
            self.baseline_predictor.fit(train_matrix)

    def fit_both(self, train_matrix: np.ndarray,
                 cosine_sim: np.ndarray,
                 pearson_sim: np.ndarray) -> None:
        """Fit với cả 2 loại similarity matrix."""
        self.train_matrix = train_matrix
        self.cosine_similarity_matrix = cosine_sim
        self.pearson_similarity_matrix = pearson_sim
        self.similarity_matrix = pearson_sim  # mặc định dùng pearson
        self._compute_user_means(train_matrix)

    def use_cosine(self) -> None:
        """Chuyển sang dùng Cosine similarity."""
        assert self.cosine_similarity_matrix is not None, "Chưa có Cosine similarity matrix. Hãy gọi fit_both()."
        self.similarity_matrix = self.cosine_similarity_matrix

    def use_pearson(self) -> None:
        """Chuyển sang dùng Pearson similarity."""
        assert self.pearson_similarity_matrix is not None, "Chưa có Pearson similarity matrix."
        self.similarity_matrix = self.pearson_similarity_matrix

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        assert self.train_matrix is not None and self.similarity_matrix is not None and self.user_means is not None, \
            "Model chưa được huấn luyện. Hãy gọi fit() trước."

        if self.train_matrix[user_idx, item_idx] > 0:
            return self.train_matrix[user_idx, item_idx]

        other_users = np.where(self.train_matrix[:, item_idx] > 0)[0]
        base_rating = self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0

        if len(other_users) == 0:
            return base_rating

        similarities = self.similarity_matrix[user_idx, other_users]
        valid_mask = similarities > 0
        if not np.any(valid_mask):
            return base_rating

        similarities = similarities[valid_mask]
        other_users = other_users[valid_mask]

        top_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        top_similarities = similarities[top_indices]
        top_other_users = other_users[top_indices]

        sim_sum = np.sum(top_similarities)
        if sim_sum == 0:
            return base_rating

        ratings = self.train_matrix[top_other_users, item_idx]

        if self.prediction_mode == 'biased_baseline' and self.baseline_predictor is not None:
            b_ui = self.baseline_predictor.predict_rating(user_idx, item_idx)
            b_vi = np.array([self.baseline_predictor.predict_rating(v, item_idx) for v in top_other_users])
            predicted_rating = b_ui + (np.sum(top_similarities * (ratings - b_vi)) / sim_sum)
        elif self.prediction_mode == 'means':
            means = self.user_means[top_other_users]
            predicted_rating = self.user_means[user_idx] + (np.sum(top_similarities * (ratings - means)) / sim_sum)
        else:
            # basic: weighted average thuần
            predicted_rating = np.sum(top_similarities * ratings) / sim_sum

        return np.clip(predicted_rating, 1.0, 5.0)

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        assert self.train_matrix is not None and self.similarity_matrix is not None and self.user_means is not None, \
            "Model chưa được huấn luyện."
        preds = np.zeros(len(item_indices))
        base_rating = self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0

        for idx, item in enumerate(item_indices):
            other_users = np.where(self.train_matrix[:, item] > 0)[0]
            if len(other_users) == 0:
                preds[idx] = base_rating
                continue

            similarities = self.similarity_matrix[user_idx, other_users]
            valid_mask = similarities > 0
            if not np.any(valid_mask):
                preds[idx] = base_rating
                continue

            similarities = similarities[valid_mask]
            other_users = other_users[valid_mask]

            top_k_idx = np.argsort(similarities)[::-1][:self.k_neighbors]
            top_sims = similarities[top_k_idx]
            top_users = other_users[top_k_idx]

            sim_sum = np.sum(top_sims)
            if sim_sum == 0:
                preds[idx] = base_rating
                continue

            item_ratings = self.train_matrix[top_users, item]

            if self.prediction_mode == 'biased_baseline' and self.baseline_predictor is not None:
                assert self.baseline_predictor.b_u is not None and self.baseline_predictor.b_i is not None
                b_ui = float(np.clip(self.baseline_predictor.mu + self.baseline_predictor.b_u[user_idx] + self.baseline_predictor.b_i[item], 1.0, 5.0))
                b_vi = np.clip(self.baseline_predictor.mu + self.baseline_predictor.b_u[top_users] + self.baseline_predictor.b_i[item], 1.0, 5.0)
                preds[idx] = b_ui + (np.sum(top_sims * (item_ratings - b_vi)) / sim_sum)
            elif self.prediction_mode == 'means':
                means = self.user_means[top_users]
                preds[idx] = self.user_means[user_idx] + (np.sum(top_sims * (item_ratings - means)) / sim_sum)
            else:
                # basic: weighted average thuần
                preds[idx] = np.sum(top_sims * item_ratings) / sim_sum

        return np.clip(preds, 1.0, 5.0)


class ItemBasedCollaborativeFiltering:
    """
    Hệ thống gợi ý Lọc cộng tác dựa trên Item (Item-based Collaborative Filtering).
    Hỗ trợ 2 loại similarity: Cosine và Adjusted Cosine.
    Prediction luôn dùng trung bình có trọng số (weighted average) thuần túy.
    """
    def __init__(self, k_neighbors: int = 40) -> None:
        self.k_neighbors = k_neighbors
        self.prediction_mode = 'basic'
        self.train_matrix: Optional[np.ndarray] = None
        # Lưu 2 similarity matrix riêng biệt
        self.cosine_similarity_matrix: Optional[np.ndarray] = None
        self.adjusted_cosine_similarity_matrix: Optional[np.ndarray] = None
        # Matrix đang được dùng (trỏ về 1 trong 2 matrix trên)
        self.similarity_matrix: Optional[np.ndarray] = None
        # Giữ lại baseline_predictor để tương thích ngược (không dùng trong prediction)
        self.baseline_predictor: Optional[BiasedPredictor] = None

    def fit(self, train_matrix: np.ndarray, similarity_matrix: np.ndarray) -> None:
        """Fit với Adjusted Cosine similarity (mặc định, tương thích ngược)."""
        self.train_matrix = train_matrix
        self.adjusted_cosine_similarity_matrix = similarity_matrix
        self.similarity_matrix = similarity_matrix  # mặc định dùng adjusted cosine

    def fit_both(self, train_matrix: np.ndarray,
                 cosine_sim: np.ndarray,
                 adjusted_cosine_sim: np.ndarray) -> None:
        """Fit với cả 2 loại similarity matrix."""
        self.train_matrix = train_matrix
        self.cosine_similarity_matrix = cosine_sim
        self.adjusted_cosine_similarity_matrix = adjusted_cosine_sim
        self.similarity_matrix = adjusted_cosine_sim  # mặc định dùng adjusted cosine

    def use_cosine(self) -> None:
        """Chuyển sang dùng Cosine similarity."""
        assert self.cosine_similarity_matrix is not None, "Chưa có Cosine similarity matrix. Hãy gọi fit_both()."
        self.similarity_matrix = self.cosine_similarity_matrix

    def use_adjusted_cosine(self) -> None:
        """Chuyển sang dùng Adjusted Cosine similarity."""
        assert self.adjusted_cosine_similarity_matrix is not None, "Chưa có Adjusted Cosine similarity matrix."
        self.similarity_matrix = self.adjusted_cosine_similarity_matrix

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        assert self.train_matrix is not None and self.similarity_matrix is not None, "Model chưa được huấn luyện."
        if self.train_matrix[user_idx, item_idx] > 0:
            return float(self.train_matrix[user_idx, item_idx])

        rated_items = np.where(self.train_matrix[user_idx, :] > 0)[0]
        if len(rated_items) == 0:
            return 3.0

        similarities = self.similarity_matrix[item_idx, rated_items]
        valid_mask = similarities > 0
        if not np.any(valid_mask):
            return 3.0

        similarities = similarities[valid_mask]
        rated_items = rated_items[valid_mask]

        top_k_idx = np.argsort(similarities)[::-1][:self.k_neighbors]
        top_sims = similarities[top_k_idx]
        top_rated_items = rated_items[top_k_idx]

        sim_sum = np.sum(top_sims)
        if sim_sum == 0:
            return 3.0

        ratings = self.train_matrix[user_idx, top_rated_items]
        # Trung bình có trọng số thuần túy (weighted average)
        predicted_rating = np.sum(top_sims * ratings) / sim_sum
        return float(np.clip(predicted_rating, 1.0, 5.0))

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        assert self.train_matrix is not None and self.similarity_matrix is not None, "Model chưa được huấn luyện."
        preds = np.zeros(len(item_indices))
        rated_items = np.where(self.train_matrix[user_idx] > 0)[0]

        if len(rated_items) == 0:
            return np.full(len(item_indices), 3.0)

        ratings = self.train_matrix[user_idx, rated_items]

        for idx, item in enumerate(item_indices):
            similarities = self.similarity_matrix[item, rated_items]
            valid_mask = similarities > 0
            if not np.any(valid_mask):
                preds[idx] = 3.0
                continue

            valid_similarities = similarities[valid_mask]
            valid_rated_items = rated_items[valid_mask]

            top_k_idx = np.argsort(valid_similarities)[::-1][:self.k_neighbors]
            top_sims = valid_similarities[top_k_idx]
            top_ratings = ratings[valid_mask][top_k_idx]

            sim_sum = np.sum(top_sims)
            # Trung bình có trọng số thuần túy (weighted average)
            preds[idx] = np.sum(top_sims * top_ratings) / sim_sum if sim_sum > 0 else 3.0

        return np.clip(preds, 1.0, 5.0)


class MatrixFactorizationSVD:
    """
    Hệ thống gợi ý dựa trên Phân rã ma trận (Matrix Factorization) sử dụng thuật toán SVD.
    Tối ưu hóa các tham số thông qua thuật toán Stochastic Gradient Descent (SGD).
    """
    def __init__(self, num_factors: int = 20, lr: float = 0.005, reg: float = 0.02, epochs: int = 20) -> None:
        self.num_factors = num_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.mu: float = 0.0
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {'epoch': [], 'train_rmse': [], 'test_rmse': []}

    def _compute_rmse(self, matrix: Optional[np.ndarray]) -> float:
        """Hàm nội bộ để tính RMSE nhanh cho lịch sử huấn luyện"""
        # Kiểm tra None đúng kiểu Optional[np.ndarray] trước khi dùng np.any
        if matrix is None or not np.any(matrix > 0): return 0.0
        if self.b_u is None or self.b_i is None or self.P is None or self.Q is None: return 0.0
        users, items = np.nonzero(matrix)
        preds = self.mu + self.b_u[users] + self.b_i[items] + np.sum(self.P[users] * self.Q[items], axis=1)
        preds = np.clip(preds, 1.0, 5.0)
        return float(np.sqrt(np.mean((matrix[users, items] - preds) ** 2)))

    def fit(self, train_matrix: np.ndarray, test_matrix: Optional[np.ndarray] = None) -> None:
        """
        Huấn luyện mô hình tìm ma trận P, Q và các hệ số bias bằng SGD.
        Bổ sung: L2 Regularization và theo dõi tiến trình huấn luyện (Overfitting).
        """
        num_users, num_items = train_matrix.shape
        nonzero_indices = np.argwhere(train_matrix > 0)
        
        self.mu = train_matrix[train_matrix > 0].mean()
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        self.P = np.random.normal(0, 0.1, (num_users, self.num_factors))
        self.Q = np.random.normal(0, 0.1, (num_items, self.num_factors))
        
        self.history = {'epoch': [], 'train_rmse': [], 'test_rmse': []}
        
        for epoch in range(self.epochs):
            np.random.shuffle(nonzero_indices)
            
            for u, i in nonzero_indices:
                r_ui = train_matrix[u, i]
                pred_r_ui = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r_ui - pred_r_ui
                
                # Cập nhật bias với L2 Regularization (trừ phần phạt)
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                # Cập nhật P và Q với L2 Regularization
                p_u = self.P[u].copy() # sao chép để dùng q_i cũ cập nhật p_u
                q_i = self.Q[i].copy() # sao chép để đảm bảo dùng giá trị cũ
                
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)

            # Đánh giá Overfitting
            train_rmse = self._compute_rmse(train_matrix)
            test_rmse = self._compute_rmse(test_matrix) if test_matrix is not None else 0.0
            
            self.history['epoch'].append(epoch + 1)
            self.history['train_rmse'].append(train_rmse)
            self.history['test_rmse'].append(test_rmse)

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        assert self.b_u is not None and self.b_i is not None and self.P is not None and self.Q is not None, \
            "Model chưa được huấn luyện. Hãy gọi fit() trước."
        pred_r_ui = self.mu + self.b_u[user_idx] + self.b_i[item_idx] + np.dot(self.P[user_idx], self.Q[item_idx])
        return float(np.clip(pred_r_ui, 1.0, 5.0))
        
    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        assert self.b_u is not None and self.b_i is not None and self.P is not None and self.Q is not None, "Model chưa được huấn luyện."
        preds = self.mu + self.b_u[user_idx] + self.b_i[item_indices] + np.dot(self.Q[item_indices], self.P[user_idx])
        return np.clip(preds, 1.0, 5.0)
    
    def save_model(self, model_path: str) -> None:
        # os.path.dirname trả về '' nếu model_path không có thư mục cha
        # → os.makedirs('') gây FileNotFoundError, cần kiểm tra trước
        parent_dir = os.path.dirname(model_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'mu': self.mu, 'b_u': self.b_u, 'b_i': self.b_i, 
                'P': self.P, 'Q': self.Q, 'num_factors': self.num_factors,
                'history': self.history
            }, f)

    def load_model(self, model_path: str) -> bool:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.mu = data['mu']
                self.b_u = data['b_u']
                self.b_i = data['b_i']
                self.P = data['P']
                self.Q = data['Q']
                self.num_factors = data['num_factors']
                self.history = data.get('history', {'epoch': [], 'train_rmse': [], 'test_rmse': []})
            return True
        return False