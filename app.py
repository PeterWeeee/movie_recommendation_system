import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Any, Dict, List

from src.data_loader import load_raw_data, load_movie_titles, load_matrix
from src.evaluation import compute_mae, compute_rmse
from src.recommender import UserBasedCollaborativeFiltering, ItemBasedCollaborativeFiltering, MatrixFactorizationSVD
from src.content_based import ContentBasedRecommender

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
processed_dir = os.path.join(base_dir, 'data', 'processed')
models_dir = os.path.join(base_dir, 'models')

@st.cache_resource
def load_system():
    if not os.path.exists(data_path) or not os.path.exists(item_path):
        return None, None, None, None, None, None, None, None
        
    df_raw = load_raw_data(data_path)
    movie_titles = load_movie_titles(item_path)
    
    train_matrix_path = os.path.join(processed_dir, 'train_matrix.npy')
    test_matrix_path = os.path.join(processed_dir, 'test_matrix.npy')
    
    if not os.path.exists(train_matrix_path) or not os.path.exists(os.path.join(models_dir, 'svd_weights.pkl')):
        st.error("Chưa tìm thấy mô hình! Vui lòng chạy lệnh `python scripts/train_pipeline.py` trước khi khởi động web.")
        return None, None, None, None, None, None, None, None
        
    train_matrix = load_matrix(train_matrix_path)
    test_matrix = load_matrix(test_matrix_path)
    
    with open(os.path.join(models_dir, 'user_cf.pkl'), 'rb') as f:
        user_cf = pickle.load(f)
        
    with open(os.path.join(models_dir, 'item_cf.pkl'), 'rb') as f:
        item_cf = pickle.load(f)
        
    svd_model = MatrixFactorizationSVD(num_factors=20, lr=0.005, reg=0.02, epochs=20)
    svd_model.load_model(os.path.join(models_dir, 'svd_weights.pkl'))
    
    with open(os.path.join(models_dir, 'content_based.pkl'), 'rb') as f:
        content_model = pickle.load(f)
        
    return df_raw, train_matrix, test_matrix, movie_titles, user_cf, item_cf, svd_model, content_model

system_objects = load_system()
if system_objects[0] is None:
    st.stop()

df_raw, train_matrix, test_matrix, movie_titles, user_cf, item_cf, svd_model, content_model = system_objects

st.title("🎬 Hệ Thống Gợi Ý Phim Lai (Hybrid Recommender)")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Trực Quan Hóa Dữ Liệu", "🎯 Gợi Ý Phim", "📈 Đánh Giá Hiệu Năng"])

# ==========================================
# TAB 1: TRỰC QUAN HÓA TRƯỚC VÀ SAU XỬ LÝ
# ==========================================
with tab1:
    st.header("Khám Phá Các Đặc Trưng Ẩn & Vấn Đề Của Hệ Thống Gợi Ý")
    
    st.subheader("1. Vấn đề cốt lõi: Độ thưa thớt và Đuôi dài (Long-Tail)")
    st.markdown("Trong thực tế, một người dùng chỉ xem một lượng rất nhỏ phim, dẫn đến ma trận dữ liệu khổng lồ nhưng lại hầu như trống rỗng. Thêm vào đó, hầu hết các đánh giá tập trung vào một số ít phim bom tấn (Phần đầu biểu đồ), trong khi hàng ngàn phim khác hiếm khi được xem (Phần đuôi).")
    
    c_prob1, c_prob2 = st.columns(2)
    with c_prob1:
        total_elements = train_matrix.shape[0] * train_matrix.shape[1]
        nonzero_elements = np.count_nonzero(train_matrix)
        sparsity = (1 - nonzero_elements / total_elements) * 100
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([nonzero_elements, total_elements - nonzero_elements], 
               labels=['Ô có dữ liệu đánh giá', 'Ô trống (Chưa xem)'], 
               autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, explode=(0.1, 0))
        ax.set_title(f"Mức Độ Thưa Thớt Của Ma Trận (Sparsity: {sparsity:.2f}%)")
        st.pyplot(fig)
        
    with c_prob2:
        fig, ax = plt.subplots(figsize=(6, 4))
        movie_rating_counts = df_raw['item_id'].value_counts().values
        ax.plot(movie_rating_counts, color='green', linewidth=2)
        ax.fill_between(range(len(movie_rating_counts)), movie_rating_counts, color='green', alpha=0.3)
        ax.set_title("Hiện tượng Đuôi Dài (Long-Tail) trong đánh giá phim")
        ax.set_xlabel("Các bộ phim (Sắp xếp theo độ phổ biến giảm dần)")
        ax.set_ylabel("Số lượng đánh giá")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("2. Mô hình SVD học được gì? (Giải mã Đặc Trưng Ẩn - Latent Features)")
    st.markdown("Matrix Factorization phân rã ma trận thưa thớt trên thành 2 ma trận nhỏ hơn: **User Features** và **Item Features**. Dưới đây, chúng ta sử dụng PCA để nén 20 chiều đặc trưng của các bộ phim xuống còn 2 chiều để trực quan hóa xem máy học đã phân cụm các bộ phim như thế nào (dù không hề biết thể loại của chúng)!")
    
    # Lấy top 100 phim phổ biến nhất để vẽ
    top_100_movies = df_raw['item_id'].value_counts().head(100).index.values
    if svd_model is not None and getattr(svd_model, 'Q', None) is not None:
        top_100_Q = svd_model.Q[top_100_movies]
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        q_pca = pca.fit_transform(top_100_Q)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(q_pca[:, 0], q_pca[:, 1], alpha=0.6, c='purple', s=50)
        
        # Hiển thị ngẫu nhiên tên của một số bộ phim trong top 100
        np.random.seed(42)
        sample_indices = np.random.choice(100, 20, replace=False)
        for idx in sample_indices:
            movie_id = top_100_movies[idx]
            movie_name = movie_titles.get(int(movie_id), f"Phim {movie_id}")
            # Rút gọn tên phim nếu quá dài
            short_name = movie_name[:20] + "..." if len(movie_name) > 20 else movie_name
            ax.annotate(short_name, (q_pca[idx, 0], q_pca[idx, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
        ax.set_title("Không Gian Đặc Trưng Ẩn 2D Của Top 100 Phim Phổ Biến (SVD + PCA)")
        ax.set_xlabel("Đặc trưng ẩn 1 (PCA Component 1)")
        ax.set_ylabel("Đặc trưng ẩn 2 (PCA Component 2)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

# ==========================================
# TAB 2: GỢI Ý PHIM
# ==========================================
with tab2:
    st.header("Cấu Hình Mục Tiêu Gợi Ý Phim")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        algo_type = st.selectbox("Họ Thuật Toán:", ["Model-Based CF", "Memory-Based CF", "Content-Based"])
    with c2:
        if algo_type == "Model-Based CF":
            algorithm = st.selectbox("Thuật toán cụ thể:", ["SVD"])
        elif algo_type == "Memory-Based CF":
            algorithm = st.selectbox("Thuật toán cụ thể:", ["User-Based KNN", "Item-Based KNN"])
        else:
            algorithm = "TF-IDF & Cosine Similarity"
            st.selectbox("Thuật toán cụ thể:", [algorithm], disabled=True)
            
    with c3:
        if algo_type == "Content-Based":
            target_id = st.number_input("Nhập mã Phim (Item_ID):", min_value=1, value=1)
        else:
            target_id = st.number_input("Nhập mã Người Dùng (User_ID):", min_value=1, value=1)
            
    with c4:
        top_n = st.slider("Số lượng (Top N):", min_value=5, max_value=20, value=5)
        
    @st.cache_data
    def generate_recommendations(t_id: int, n: int, a_type: str, a_name: str) -> Tuple[list, str]:
        # Xử lý Content-Based
        if a_type == "Content-Based":
            try:
                recs = content_model.get_content_based_recommendations(t_id, n)
                if not recs:
                    return [], "Không tìm thấy phim này trong cơ sở dữ liệu để tìm phim tương tự."
                rec_list = [{"Hạng": r, "Tên Bộ Phim": rec[1], "Điểm Dự Kiến": f"Độ tương đồng: {rec[2]:.2f}"} for r, rec in enumerate(recs, 1)]
                return rec_list, "Content-Based"
            except Exception as e:
                return [], str(e)
                
        # Xử lý Lọc cộng tác (CF)
        num_users = train_matrix.shape[0]
        if t_id > num_users: # Cold Start for User
            popular_movie_ids = df_raw['item_id'].value_counts().head(n).index
            rec_list = []
            for rank, m_id in enumerate(popular_movie_ids, 1):
                movie_name = movie_titles.get(int(m_id), f"Phim ID {int(m_id)}")
                rec_list.append({"Hạng": rank, "Tên Bộ Phim": movie_name, "Điểm Dự Kiến": "Phổ biến nhất"})
            return rec_list, "Popularity-Based (Cold Start Fallback)"

        user_idx = t_id - 1
        if a_name == "SVD":
            model = svd_model
        elif a_name == "User-Based KNN":
            model = user_cf
        else:
            model = item_cf
        
        unviewed_items = np.where(train_matrix[user_idx] == 0)[0]
        if len(unviewed_items) == 0:
            return [], "Người dùng này đã xem tất cả các phim!"
            
        # Sử dụng Vectorization để tính cực nhanh
        preds = model.predict_batch(user_idx, unviewed_items)
        
        top_n_indices = np.argsort(preds)[-n:][::-1]
        top_item_indices = unviewed_items[top_n_indices]
        top_scores = preds[top_n_indices]
        
        rec_list = []
        for rank, (item_idx, score) in enumerate(zip(top_item_indices, top_scores), 1):
            movie_name = movie_titles.get(int(item_idx) + 1, f"Phim ID {int(item_idx) + 1}")
            rec_list.append({
                "Hạng": rank,
                "Tên Bộ Phim": movie_name,
                "Điểm Dự Kiến": f"{score:.2f} ⭐"
            })
        return rec_list, a_name

    if st.button("🚀 Phát Sinh Danh Sách Gợi Ý"):
        rec_list, applied_algo = generate_recommendations(target_id, top_n, algo_type, algorithm)
        if "Cold Start" in applied_algo:
            st.warning("⚠️ Người dùng mới (Cold Start) - Hệ thống tự động lùi về chế độ Gợi ý Phim Phổ biến (Popularity-Based)")
        else:
            if algo_type == "Content-Based":
                movie_name = movie_titles.get(target_id, f"Phim {target_id}")
                st.success(f"Kết quả tìm kiếm phim có nội dung tương tự với **{movie_name}**:")
            else:
                st.success(f"Kết quả gợi ý Top {top_n} phim cho **User_ID {target_id}** bằng thuật toán **{applied_algo}**:")
        
        if rec_list:
            st.table(pd.DataFrame(rec_list).set_index("Hạng"))

# ==========================================
# TAB 3: ĐÁNH GIÁ HIỆU NĂNG
# ==========================================
with tab3:
    st.header("So Sánh Độ Lệch Sai Số Giữa Các Thuật Toán Lọc Cộng Tác")
    c_eval1, c_eval2 = st.columns(2)
    
    with c_eval1:
        if svd_model and len(svd_model.history.get('epoch', [])) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            epochs = svd_model.history['epoch']
            train_rmse = svd_model.history['train_rmse']
            test_rmse = svd_model.history['test_rmse']
            
            ax.plot(epochs, train_rmse, label='Train RMSE', marker='o', color='blue')
            ax.plot(epochs, test_rmse, label='Test RMSE', marker='s', color='red')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('RMSE')
            ax.set_title("Quá trình huấn luyện SVD (Overfitting Tracking)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.info("Chưa có dữ liệu lịch sử huấn luyện SVD trong file weights để vẽ biểu đồ.")
            
    with c_eval2:
        if st.button("📊 Chạy So Sánh Kép Hệ Thống"):
            with st.spinner("Đang tính toán sai số trên tập Test Set..."):
                user_cf_mae = compute_mae(test_matrix, user_cf)
                user_cf_rmse = compute_rmse(test_matrix, user_cf)
                item_cf_mae = compute_mae(test_matrix, item_cf)
                item_cf_rmse = compute_rmse(test_matrix, item_cf)
                svd_mae = compute_mae(test_matrix, svd_model)
                svd_rmse = compute_rmse(test_matrix, svd_model)
                
                eval_data = {
                    'Thuật toán': ['User-Based CF', 'User-Based CF', 
                                   'Item-Based CF', 'Item-Based CF',
                                   'SVD', 'SVD'],
                    'Độ đo': ['MAE', 'RMSE', 'MAE', 'RMSE', 'MAE', 'RMSE'],
                    'Giá trị': [user_cf_mae, user_cf_rmse, item_cf_mae, item_cf_rmse, svd_mae, svd_rmse]
                }
                eval_df = pd.DataFrame(eval_data)
                
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.barplot(x='Thuật toán', y='Giá trị', hue='Độ đo', data=eval_df, palette='Set2', ax=ax)
                ax.set_title("Biểu đồ So sánh Hiệu suất (Benchmark Chart)")
                ax.set_ylabel("Giá trị sai số (Càng thấp càng tốt)")
                st.pyplot(fig)
                
                summary_df = pd.DataFrame({
                    'Thuật toán': ['User-Based CF', 'Item-Based CF', 'Matrix Factorization (SVD)'],
                    'MAE': [f"{user_cf_mae:.4f}", f"{item_cf_mae:.4f}", f"{svd_mae:.4f}"],
                    'RMSE': [f"{user_cf_rmse:.4f}", f"{item_cf_rmse:.4f}", f"{svd_rmse:.4f}"]
                })
                st.dataframe(summary_df, use_container_width=True)