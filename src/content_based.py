import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

def load_item_metadata(file_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file u.item và trích xuất các thông tin cần thiết cho Content-Based.
    Kết hợp tên phim và các thể loại thành một chuỗi văn bản (content_string) để dùng TF-IDF.
    """
    genres = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    # Định nghĩa tên các cột tương ứng với cấu trúc file u.item
    columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + genres
    
    # Đọc file với mã hóa ISO-8859-1
    df = pd.read_csv(file_path, sep='|', names=columns, encoding='ISO-8859-1')
    
    # Tạo cột 'content_string' bằng cách ghép tên phim và các thể loại mà phim đó có
    content_strings = []
    for _, row in df.iterrows():
        # Lấy tên phim, bỏ đi phần năm (nếu muốn) hoặc giữ nguyên
        title = str(row['movie_title'])
        
        # Lọc ra các thể loại có giá trị 1
        active_genres = [g for g in genres if row[g] == 1]
        
        # Ghép lại thành một chuỗi
        content_str = title + " " + " ".join(active_genres)
        content_strings.append(content_str)
        
    df['content_string'] = content_strings
    return df

class ContentBasedRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung (Content-Based Filtering).
    Sử dụng TF-IDF Vectorizer và Cosine Similarity.
    """
    def __init__(self):
        self.df = None
        self.cosine_sim_matrix = None
        self.indices = None
        
    def fit(self, df: pd.DataFrame):
        """
        Huấn luyện mô hình: Tính toán ma trận TF-IDF và Cosine Similarity
        """
        self.df = df
        
        # 1. Khởi tạo TF-IDF Vectorizer
        # Có thể dùng stop_words='english' để loại bỏ các từ vô nghĩa trong tên phim
        tfidf = TfidfVectorizer(stop_words='english')
        
        # 2. Xây dựng ma trận TF-IDF từ cột content_string
        tfidf_matrix = tfidf.fit_transform(df['content_string'])
        
        # 3. Tính toán ma trận độ tương đồng Cosine
        self.cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 4. Tạo một Series để tra cứu chỉ số của bộ phim (index) thông qua movie_id
        # Mảng numpy thường tính index từ 0, trong khi movie_id từ 1
        self.indices = pd.Series(df.index, index=df['movie_id'])
        
    def get_content_based_recommendations(self, movie_id: int, top_n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Gợi ý các bộ phim tương đồng dựa trên nội dung.
        Dùng làm phương án dự phòng khi gặp lỗi Khởi động lạnh (Cold Start).
        
        Trả về: Danh sách các tuple dạng (movie_id, movie_title, độ_tương_đồng)
        """
        if self.cosine_sim_matrix is None or self.df is None or self.indices is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi fit() trước.")
            
        if movie_id not in self.indices:
            return []
            
        # Lấy chỉ số (index) của bộ phim
        idx = self.indices[movie_id]
        
        # Lấy mảng độ tương đồng của bộ phim này với tất cả các phim khác
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))
        
        # Sắp xếp các phim dựa trên điểm tương đồng giảm dần
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Lấy top_n phim tương đồng nhất (bỏ qua phim ở vị trí 0 vì đó chính là nó)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        top_scores = [i[1] for i in sim_scores[1:top_n+1]]
        
        # Trả về kết quả
        recommendations = []
        for i, score in zip(top_indices, top_scores):
            rec_id = self.df.iloc[i]['movie_id']
            rec_title = self.df.iloc[i]['movie_title']
            recommendations.append((rec_id, rec_title, score))
            
        return recommendations
