import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sqlalchemy import create_engine, text
import urllib
import pyodbc

USE_SQL_SERVER = True
DB_NAME = 'MovieRecommendationDB'
SERVER = 'localhost'

def get_engine():
    drivers = [d for d in pyodbc.drivers()]
    driver = 'SQL Server'
    if 'ODBC Driver 17 for SQL Server' in drivers:
        driver = 'ODBC Driver 17 for SQL Server'
    elif 'ODBC Driver 18 for SQL Server' in drivers:
        driver = 'ODBC Driver 18 for SQL Server'
        
    params = urllib.parse.quote_plus(
        f'DRIVER={{{driver}}};'
        f'SERVER={SERVER};'
        f'DATABASE={DB_NAME};'
        f'Trusted_Connection=yes;'
        f'TrustServerCertificate=yes;'
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def load_raw_data(file_path: str) -> pd.DataFrame:
    if USE_SQL_SERVER:
        try:
            engine = get_engine()
            query = "SELECT user_id, movie_id as item_id, rating, timestamp FROM ratings"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Failed to load from SQL Server: {e}. Falling back to CSV.")
            
    # Đọc tệp dữ liệu u.data với phân tách bằng dấu tab
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=columns)
    return df

def build_user_item_matrix(df: pd.DataFrame) -> np.ndarray:
    # Xác định số lượng user và item lớn nhất để làm kích thước ma trận
    num_users = df['user_id'].max()
    num_items = df['item_id'].max()
    
    # Khởi tạo ma trận toàn số 0 với kích thước (num_users x num_items)
    # Dùng kiểu dữ liệu float để thuận tiện cho các phép chia toán học
    matrix = np.zeros((num_users, num_items), dtype=np.float64)
    
    # Điền giá trị rating vào ma trận
    # Trừ 1 ở chỉ số dòng và cột vì ID trong file bắt đầu từ 1, còn chỉ số mảng bắt đầu từ 0
    for row in df.itertuples():
        matrix[row.user_id - 1, row.item_id - 1] = row.rating
    return matrix

def train_test_split_matrix(matrix: np.ndarray, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    train_matrix = matrix.copy()
    test_matrix = np.zeros_like(matrix)
    
    # Tìm tọa độ của tất cả các ô đã được đánh giá (giá trị > 0)
    nonzero_positions = np.argwhere(matrix > 0)
    
    # Xác định số lượng mẫu dữ liệu sẽ đưa vào tập test
    num_nonzero = len(nonzero_positions)
    num_test = int(num_nonzero * test_ratio)
    
    # Lựa chọn ngẫu nhiên các chỉ số tọa độ để đưa vào tập test
    test_indices = np.random.choice(num_nonzero, size=num_test, replace=False)
    
    # Chuyển đổi dữ liệu từ ma trận gốc sang hai ma trận Train và Test
    for idx in test_indices:
        u, i = nonzero_positions[idx]
        test_matrix[u, i] = matrix[u, i]
        train_matrix[u, i] = 0.0
        
    return train_matrix, test_matrix

def save_matrix(matrix: np.ndarray, target_path: str) -> None:
    # Dòng này đảm bảo tự tạo các thư mục cha (như data/processed) nếu chưa có
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    np.save(target_path, matrix)

def load_matrix(target_path: str) -> np.ndarray:
    return np.load(target_path)

def load_movie_titles(file_path: str) -> Dict[int, str]:
    """
    Đọc tệp u.item để lấy bản đồ ánh xạ từ item_id sang tên bộ phim.
    """
    if USE_SQL_SERVER:
        try:
            engine = get_engine()
            query = "SELECT movie_id, title FROM movies"
            df = pd.read_sql(query, engine)
            return dict(zip(df['movie_id'], df['title']))
        except Exception as e:
            print(f"Failed to load movies from SQL Server: {e}. Falling back to CSV.")

    movie_titles = {}
    # Sử dụng mã hóa ISO-8859-1 vì tệp u.item chứa một số ký tự đặc biệt không thuộc UTF-8
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            fields = line.strip().split('|')
            if len(fields) > 1:
                item_id = int(fields[0])
                movie_title = fields[1]
                movie_titles[item_id] = movie_title
    return movie_titles

def get_max_user_id() -> int:
    if not USE_SQL_SERVER:
        return 943
    try:
        engine = get_engine()
        with engine.begin() as conn:
            res = conn.execute(text("SELECT ISNULL(MAX(user_id), 0) FROM users"))
            return res.scalar()
    except Exception as e:
        print(f"Failed to get max user id from SQL Server: {e}")
        return 943

def add_new_user(age: int, gender: str, occupation: str, zip_code: str) -> int:
    if not USE_SQL_SERVER:
        raise NotImplementedError("SQL Server is not enabled.")
    engine = get_engine()
    with engine.begin() as conn:
        res = conn.execute(text("SELECT ISNULL(MAX(user_id), 0) FROM users"))
        new_id = res.scalar() + 1
        query = text("INSERT INTO users (user_id, age, gender, occupation, zip_code) VALUES (:id, :a, :g, :o, :z)")
        conn.execute(query, {"id": new_id, "a": age, "g": gender, "o": occupation, "z": zip_code})
        return new_id

def add_movie_rating(user_id: int, item_id: int, rating: int):
    if not USE_SQL_SERVER:
        raise NotImplementedError("SQL Server is not enabled.")
    engine = get_engine()
    import time
    timestamp = int(time.time())
    with engine.begin() as conn:
        query = text("INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES (:u, :m, :r, :t)")
        conn.execute(query, {"u": user_id, "m": item_id, "r": rating, "t": timestamp})

def get_or_create_processed_matrices(data_path: str, processed_dir: str, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hàm thông minh: Tự động kiểm tra file đã xử lý cũ. 
    Nếu có thì nạp lên ngay, nếu chưa có thì mới tính toán rồi lưu lại vào data/processed/
    """
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, 'train_matrix.npy')
    test_path = os.path.join(processed_dir, 'test_matrix.npy')
    
    # Nếu đã từng tính toán và lưu file trước đó, nạp trực tiếp để tiết kiệm thời gian
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_matrix = np.load(train_path)
        test_matrix = np.load(test_path)
        return train_matrix, test_matrix
    
    # Nếu chưa có, tiến hành xử lý từ file gốc u.data
    df = load_raw_data(data_path)
    full_matrix = build_user_item_matrix(df)
    train_matrix, test_matrix = train_test_split_matrix(full_matrix, test_ratio=test_ratio)
    
    # Lưu lại để lần sau sử dụng
    np.save(train_path, train_matrix)
    np.save(test_path, test_matrix)
    return train_matrix, test_matrix