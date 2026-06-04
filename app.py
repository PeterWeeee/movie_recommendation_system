import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Any, Dict

from src.data_loader import load_raw_data, load_movie_titles, get_or_create_processed_matrices, build_user_item_matrix
from src.similarity import compute_pearson_similarity
from src.recommender import UserBasedCollaborativeFiltering, MatrixFactorizationSVD
from src.evaluation import compute_mae, compute_rmse

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
processed_dir = os.path.join(base_dir, 'data', 'processed')
model_path = os.path.join(base_dir, 'models', 'svd_weights.pkl')

@st.cache_resource
def load_and_train_system() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[int, str]], Optional[Any], Optional[Any]]:
    if not os.path.exists(data_path) or not os.path.exists(item_path):
        return None, None, None, None, None, None
        
    df_raw = load_raw_data(data_path)
    movie_titles = load_movie_titles(item_path)
    
    # 1. Tận dụng thư mục data/processed/ để đọc ma trận Train/Test tối ưu
    train_matrix, test_matrix = get_or_create_processed_matrices(data_path, processed_dir)
    
    # 2. Tạo ma trận độ tương đồng cho User-Based
    pearson_sim = compute_pearson_similarity(train_matrix)
    user_cf = UserBasedCollaborativeFiltering(k_neighbors=40)
    user_cf.fit(train_matrix, pearson_sim)
    
    # 3. Tận dụng thư mục models/ để quản lý trọng số mô hình SVD
    svd_model = MatrixFactorizationSVD(num_factors=20, lr=0.005, reg=0.02, epochs=20)
    if os.path.exists(model_path):
        svd_model.load_model(model_path)
    else:
        svd_model.fit(train_matrix)
        svd_model.save_model(model_path) # Lưu lại sau khi học xong
        
    return df_raw, train_matrix, test_matrix, movie_titles, user_cf, svd_model

df_raw, train_matrix, test_matrix, movie_titles, user_cf, svd_model = load_and_train_system()

if df_raw is None or train_matrix is None or test_matrix is None or movie_titles is None:
    st.error(f"Thiếu tệp dữ liệu trong thư mục data/raw/! Vui lòng đảm bảo các file 'u.data' và 'u.item' tồn tại trong {os.path.join(base_dir, 'data', 'raw')}.")
else:
    st.title("🎬 Hệ Thống Gợi Ý Phim Chuyên Nghiệp")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Trực Quan Hóa Dữ Liệu (Trước & Sau)", "🎯 Gợi Ý Phim Cá Nhân", "📈 Đánh Giá Hiệu Năng"])
    
    # ==========================================
    # TAB 1: TRỰC QUAN HÓA TRƯỚC VÀ SAU XỬ LÝ
    # ==========================================
    with tab1:
        st.header("Trực Quan Hóa Trạng Thái Dữ Liệu Ở Các Giai Đoạn")
        
        st.subheader("1. Giai đoạn dữ liệu thô ban đầu (Trước xử lý)")
        
        @st.cache_data
        def get_user_counts(df: pd.DataFrame) -> pd.Series:
            return df['user_id'].value_counts()
            
        c_raw1, c_raw2 = st.columns(2)
        with c_raw1:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.countplot(x='rating', data=df_raw, palette='Blues_r', ax=ax)
            ax.set_title("Phân Phối Các Mức Điểm Đánh Giá Gốc")
            st.pyplot(fig)
        with c_raw2:
            fig, ax = plt.subplots(figsize=(6, 3))
            # Biểu diễn mật độ xem phim theo độ tuổi của tệp u.user (Ví dụ phân tích mở rộng dựa trên lượng rating)
            user_counts = get_user_counts(df_raw)
            sns.histplot(user_counts, bins=30, kde=True, color='purple', ax=ax)
            ax.set_title("Phân Bố Số Lượt Đánh Giá Trên Mỗi Người Dùng")
            ax.set_xlabel("Số lượt đánh giá")
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("2. Giai đoạn cấu trúc ma trận và mô hình hóa (Sau tiền xử lý)")
        c_proc1, c_proc2 = st.columns(2)
        with c_proc1:
            # Tính toán độ thưa thớt của ma trận kề
            total_elements = train_matrix.shape[0] * train_matrix.shape[1]
            nonzero_elements = np.count_nonzero(train_matrix)
            sparsity = (1 - nonzero_elements / total_elements) * 100
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.pie([nonzero_elements, total_elements - nonzero_elements], 
                   labels=['Ô có điểm', 'Ô trống (0)'], 
                   autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
            ax.set_title(f"Mức Độ Thưa Thớt Của Ma Trận Kề: {sparsity:.2f}%")
            st.pyplot(fig)
            
        with c_proc2:
            # Mô phỏng một phần nhỏ của ma trận tương đồng sau khi chạy thuật toán
            fig, ax = plt.subplots(figsize=(6, 3))
            # Trích xuất ma trận tương đồng kích thước 10x10 của 10 user đầu tiên để minh họa Heatmap
            sample_sim = compute_pearson_similarity(train_matrix[:10, :])
            sns.heatmap(sample_sim, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=False)
            ax.set_title("Heatmap Trực Quan Ma Trận Tương Đồng (10 Users Đầu Tiên)")
            st.pyplot(fig)

    # ==========================================
    # TAB 2: GỢI Ý PHIM
    # ==========================================
    with tab2:
        st.header("Cấu Hình Mục Tiêu Gợi Ý Phim")
        c1, c2, c3 = st.columns(3)
        with c1:
            user_id = st.number_input("Nhập mã User_ID cần gợi ý (1 - 943):", min_value=1, max_value=943, value=1)
        with c2:
            top_n = st.slider("Số lượng phim muốn gợi ý (Top N):", min_value=5, max_value=20, value=5)
        with c3:
            algorithm = st.selectbox("Lựa chọn thuật toán AI:", ["Matrix Factorization (SVD)", "User-Based CF (KNN)"])
            
        if st.button("🚀 Phát Sinh Danh Sách Gợi Ý"):
            user_idx = user_id - 1
            st.markdown(f"### Kết quả gợi ý Top {top_n} phim cho **User_ID {user_id}** bằng thuật toán **{algorithm}**:")
            
            model = svd_model if algorithm == "Matrix Factorization (SVD)" else user_cf
            if model is None:
                st.error("Mô hình chưa được khởi tạo. Vui lòng kiểm tra dữ liệu đầu vào.")
                st.stop()
            unviewed_items = np.where(train_matrix[user_idx] == 0)[0]
            
            predicted_ratings = []
            for item_idx in unviewed_items:
                pred_score = model.predict_rating(user_idx, item_idx)
                predicted_ratings.append((item_idx, pred_score))
                
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = predicted_ratings[:top_n]
            
            rec_list = []
            for rank, (item_idx, score) in enumerate(top_recommendations, 1):
                movie_name = movie_titles.get(int(item_idx) + 1, f"Phim ID {int(item_idx) + 1}")
                rec_list.append({
                    "Hạng": rank,
                    "Tên Bộ Phim": movie_name,
                    "Điểm Dự Kiến Chấm (Sao)": f"{score:.2f} ⭐"
                })
            st.table(pd.DataFrame(rec_list).set_index("Hạng"))

    # ==========================================
    # TAB 3: ĐÁNH GIÁ HIỆU NĂNG
    # ==========================================
    with tab3:
        st.header("So Sánh Độ Lệch Sai Số Giữa Các Thuật Toán")
        if st.button("📊 Chạy Kiểm Thử Toàn Diện Hệ Thống"):
            with st.spinner("Đang tính toán sai số trên tập dữ liệu kiểm thử (Test Set)..."):
                user_cf_mae = compute_mae(test_matrix, user_cf)
                user_cf_rmse = compute_rmse(test_matrix, user_cf)
                svd_mae = compute_mae(test_matrix, svd_model)
                svd_rmse = compute_rmse(test_matrix, svd_model)
                
                eval_data = {
                    'Thuật toán': ['User-Based CF', 'User-Based CF', 'SVD (SGD)', 'SVD (SGD)'],
                    'Độ đo sai số': ['MAE', 'RMSE', 'MAE', 'RMSE'],
                    'Giá trị': [user_cf_mae, user_cf_rmse, svd_mae, svd_rmse]
                }
                eval_df = pd.DataFrame(eval_data)
                
                fig, ax = plt.subplots(figsize=(7, 3.5))
                sns.barplot(x='Độ đo sai số', y='Giá trị', hue='Thuật toán', data=eval_df, palette='Set2', ax=ax)
                ax.set_ylabel("Giá trị sai số")
                st.pyplot(fig)
                
                summary_df = pd.DataFrame({
                    'Thuật toán cài đặt': ['User-Based CF (Pearson)', 'Matrix Factorization (SVD)'],
                    'MAE': [f"{user_cf_mae:.4f}", f"{svd_mae:.4f}"],
                    'RMSE': [f"{user_cf_rmse:.4f}", f"{svd_rmse:.4f}"]
                })
                st.dataframe(summary_df, use_container_width=True)