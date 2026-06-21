import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from typing import Tuple, Optional, Any, Dict, List

from src.data_loader import load_raw_data, load_movie_titles, load_matrix, add_new_user, add_movie_rating, get_max_user_id
from src.evaluation import compute_mae, compute_rmse
from src.recommender import UserBasedCollaborativeFiltering, ItemBasedCollaborativeFiltering, MatrixFactorizationSVD
from src.content_based import ContentBasedRecommender

st.set_page_config(page_title="MovieFlix - Đề Xuất Phim", layout="wide", page_icon="🍿")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
processed_dir = os.path.join(base_dir, 'data', 'processed')
models_dir = os.path.join(base_dir, 'models')

# ==========================================
# GIAO DIỆN CSS (Movie Cards)
# ==========================================
st.markdown("""
<style>
.movie-card {
    background-color: #1e1e1e;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.5);
}
.movie-poster-container {
    width: 100%;
    padding-top: 150%; /* Aspect ratio 2:3 for posters */
    position: relative;
    border-radius: 5px;
    overflow: hidden;
    background-color: #2b2b2b;
}
.movie-poster {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.movie-title {
    color: white;
    font-size: 15px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 5px;
    line-height: 1.2;
    flex-grow: 1;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-rating {
    color: #f5c518;
    font-size: 14px;
    font-weight: bold;
}
.movie-rank {
    position: absolute;
    top: 5px;
    left: 5px;
    background-color: rgba(229, 9, 20, 0.9);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    z-index: 10;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CÁC HÀM XỬ LÝ (Backend)
# ==========================================
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

@st.cache_data(show_spinner=False)
def fetch_poster(title: str, api_key: str = "") -> str:
    """Gọi TMDB API để lấy ảnh bìa. Nếu không có key, dùng ảnh mặc định."""
    clean_title = title.split("(")[0].strip()
    if not api_key:
        return f"https://via.placeholder.com/300x450/2b2b2b/FFFFFF?text={clean_title.replace(' ', '+')}"
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and data['results'][0].get('poster_path'):
                return "https://image.tmdb.org/t/p/w500/" + data['results'][0]['poster_path']
    except Exception as e:
        pass
    
    return f"https://via.placeholder.com/300x450/2b2b2b/FFFFFF?text={clean_title.replace(' ', '+')}"

def render_movie_grid(movies_list: List[Dict], api_key: str, cols_count: int = 5):
    """Hiển thị danh sách phim dưới dạng lưới các thẻ."""
    for i in range(0, len(movies_list), cols_count):
        cols = st.columns(cols_count)
        for j in range(cols_count):
            if i + j < len(movies_list):
                movie = movies_list[i + j]
                poster_url = fetch_poster(movie['title'], api_key)
                
                rank_badge = ""
                if 'rank' in movie:
                    rank_badge = f'<div class="movie-rank">#{movie["rank"]}</div>'
                
                import html
                safe_title = html.escape(movie['title'])
                
                with cols[j]:
                    html_content = (
                        '<div class="movie-card">'
                        '<div class="movie-poster-container">'
                        f'{rank_badge}'
                        f'<img src="{poster_url}" class="movie-poster" alt="{safe_title}">'
                        '</div>'
                        f'<div class="movie-title" title="{safe_title}">{safe_title}</div>'
                        f'<div class="movie-rating">{movie["score"]}</div>'
                        '</div>'
                    )
                    st.markdown(html_content, unsafe_allow_html=True)

# Khởi tạo hệ thống
system_objects = load_system()
if system_objects[0] is None:
    st.stop()

df_raw, train_matrix, test_matrix, movie_titles, user_cf, item_cf, svd_model, content_model = system_objects

# Tính toán top phổ biến (tính toán sẵn một lần)
@st.cache_data
def get_popular_movies(top_n=10):
    # Sử dụng thuật toán Biased Predictor thay vì đếm số lượng thông thường
    # Lấy model BiasedPredictor đã được huấn luyện bên trong item_cf
    baseline = item_cf.baseline_predictor
    
    # Lọc ra các phim có ít nhất 50 lượt đánh giá để tránh các phim "ảo" (chỉ 1 người đánh giá 5 sao)
    item_counts = df_raw['item_id'].value_counts()
    valid_items = item_counts[item_counts > 50].index
    
    if baseline is not None and baseline.b_i is not None:
        item_scores = []
        for m_id in valid_items:
            idx = int(m_id) - 1 # Chuyển ID thành index
            if idx < len(baseline.b_i):
                # Công thức Biased Predictor: Điểm phim = mu (trung bình hệ thống) + b_i (độ lệch của phim)
                score = baseline.mu + baseline.b_i[idx]
                item_scores.append((m_id, score))
                
        # Sắp xếp lấy phim có điểm (Bias) cao nhất
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:top_n]
        
        res = []
        for rank, (m_id, score) in enumerate(top_items, 1):
            name = movie_titles.get(int(m_id), f"Phim {int(m_id)}")
            res.append({"title": name, "score": f"Chất lượng: {score:.2f} ⭐", "rank": rank})
        return res
    else:
        # Fallback về phương pháp cũ nếu không tìm thấy mô hình Biased Predictor
        popular_ids = item_counts.head(top_n).index
        res = []
        for rank, m_id in enumerate(popular_ids, 1):
            name = movie_titles.get(int(m_id), f"Phim {int(m_id)}")
            res.append({"title": name, "score": "Đang thịnh hành 🔥", "rank": rank})
        return res

@st.cache_data
def get_svd_recommendations(user_id: int, top_n=10):
    user_idx = user_id - 1
    num_users = train_matrix.shape[0]
    if user_idx >= num_users:
        return [] # Cold start
        
    unviewed_items = np.where(train_matrix[user_idx] == 0)[0]
    if len(unviewed_items) == 0:
        return []
        
    preds = svd_model.predict_batch(user_idx, unviewed_items)
    top_n_indices = np.argsort(preds)[-top_n:][::-1]
    
    top_item_indices = unviewed_items[top_n_indices]
    top_scores = preds[top_n_indices]
    
    res = []
    for rank, (item_idx, score) in enumerate(zip(top_item_indices, top_scores), 1):
        name = movie_titles.get(int(item_idx) + 1, f"Phim {int(item_idx) + 1}")
        res.append({"title": name, "score": f"{score:.2f} ⭐", "rank": rank})
    return res

@st.cache_data
def get_similar_movies(movie_id: int, top_n=5):
    try:
        movie_idx = movie_id - 1
        # Chuyển sang dùng Item-Based CF để tìm phim tương tự (Collaborative Filtering thuần)
        similarities = item_cf.similarity_matrix[movie_idx]
        
        # Sắp xếp độ tương đồng giảm dần
        top_indices = np.argsort(similarities)[::-1]
        
        res = []
        rank = 1
        for idx in top_indices:
            # Bỏ qua chính bộ phim đó
            if idx == movie_idx:
                continue
            # Bỏ qua nếu độ tương đồng <= 0 (không có dữ liệu đánh giá chung)
            if similarities[idx] <= 0:
                continue
                
            name = movie_titles.get(int(idx) + 1, f"Phim {int(idx) + 1}")
            res.append({"title": name, "score": f"Độ tương đồng: {similarities[idx]:.2f}", "rank": rank})
            rank += 1
            if len(res) == top_n:
                break
        return res
    except Exception as e:
        return []

def render_expandable_grid(movies_list: List[Dict], api_key: str, section_key: str, default_count: int = 5):
    """Hàm hỗ trợ hiển thị lưới phim có tính năng Xem thêm / Thu gọn."""
    state_key = f"expand_{section_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_count

    current_count = st.session_state[state_key]
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if current_count == default_count and len(movies_list) > default_count:
            if st.button("Xem thêm", key=f"btn_{section_key}", use_container_width=True):
                st.session_state[state_key] = current_count + 5
                st.rerun()
        elif current_count > default_count:
            if st.button("Thu gọn", key=f"btn_collapse_{section_key}", use_container_width=True):
                st.session_state[state_key] = default_count
                st.rerun()
                
    render_movie_grid(movies_list[:current_count], api_key, cols_count=5)
    
    if current_count > default_count and current_count < len(movies_list):
        if st.button("Tải thêm 5 bộ phim", key=f"btn_loadmore_{section_key}"):
            st.session_state[state_key] += 5
            st.rerun()

# ==========================================
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# ==========================================
st.sidebar.title("🍿 MovieFlix")

page = st.sidebar.radio("Điều Hướng", ["Trang Chủ (Khám Phá)", "Đánh giá của người dùng", "Dành Cho Developer"])
st.sidebar.markdown("---")

st.sidebar.header("Cài Đặt Hệ Thống")
tmdb_key = st.sidebar.text_input("TMDB API Key (tùy chọn)", value="138b5cfdd869964ea7c06f50d783803a", type="password", help="Nhập key TMDB để tải ảnh bìa phim thực tế. Nếu để trống sẽ dùng ảnh mặc định.")

# Đăng nhập mô phỏng
st.sidebar.markdown("---")
st.sidebar.header("👤 Đăng nhập")

if 'max_user_id' not in st.session_state:
    st.session_state.max_user_id = max(train_matrix.shape[0], get_max_user_id())

current_user = st.sidebar.number_input(f"Nhập User ID (1 - {st.session_state.max_user_id})", min_value=1, max_value=st.session_state.max_user_id, value=1)
st.sidebar.success(f"Đã đăng nhập với tư cách **User {current_user}**")

st.sidebar.markdown("---")
st.sidebar.subheader("Tạo User Mới")
with st.sidebar.form("new_user_form"):
    age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Giới tính", ["M", "F"])
    occupation = st.text_input("Nghề nghiệp", value="student")
    zip_code = st.text_input("Mã bưu điện", value="00000")
    submit_user = st.form_submit_button("Tạo User")
    
    if submit_user:
        try:
            new_id = add_new_user(age, gender, occupation, zip_code)
            st.session_state.max_user_id = new_id
            st.success(f"Tạo thành công! User ID mới là: {new_id}")
            # Xóa cache để cập nhật
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: {e}")


# ==========================================
# TRANG CHỦ (Home Page)
# ==========================================
if page == "Trang Chủ (Khám Phá)":
    
    # 1. Phim Dành Cho Bạn (For You)
    st.header(f"✨ Dành Cho Bạn, User {current_user}")
    st.markdown("*(Số sao hiển thị bên dưới là điểm số **dự đoán** mà thuật toán SVD nghĩ bạn sẽ chấm cho phim này)*")

    # Lấy thêm để có thể mở rộng (lấy top 20)
    user_recs = get_svd_recommendations(current_user, top_n=20)
    
    if not user_recs:
        st.info("Chào người dùng mới! Đây là lần đầu bạn đến với hệ thống, hãy xem các phim thịnh hành bên dưới nhé.")
    else:
        render_expandable_grid(user_recs, tmdb_key, "foryou")
        
    st.markdown("---")
    
    # 2. Phim Thịnh Hành (Trending)
    st.header("🔥 Phim Đang Thịnh Hành")
    popular_movies = get_popular_movies(top_n=20)
    render_expandable_grid(popular_movies, tmdb_key, "trending")
        
    st.markdown("---")
    
    # 3. Khám Phá & Phim Tương Tự
    st.header("🔍 Khám Phá Phim")
    
    # Tạo danh sách các phim cho ô selectbox
    movie_options = [f"{id} - {title}" for id, title in movie_titles.items()]
    selected_movie_str = st.selectbox("Tìm kiếm phim bạn yêu thích để xem các phim tương tự:", movie_options)
    
    if selected_movie_str:
        selected_id = int(selected_movie_str.split(" - ")[0])
        st.subheader("Khán giả xem phim này cũng thích:")
        similar_movies = get_similar_movies(selected_id, top_n=20)
        if similar_movies:
            render_expandable_grid(similar_movies, tmdb_key, "similar")
        else:
            st.warning("Xin lỗi, chưa có đủ dữ liệu để gợi ý phim tương tự cho bộ phim này.")

# ==========================================
# ĐÁNH GIÁ CỦA NGƯỜI DÙNG
# ==========================================
elif page == "Đánh giá của người dùng":
    st.title("👤 Đánh giá của người dùng")
    
    tab_history, tab_rating = st.tabs(["🕒 Lịch sử đánh giá", "⭐ Đánh giá film"])
    
    with tab_history:
        st.subheader(f"Lịch sử đánh giá của User {current_user}")
        user_idx = current_user - 1
        
        # Check if user_idx is valid
        if user_idx >= train_matrix.shape[0]:
            st.info("Người dùng này mới tạo, chưa có lịch sử đánh giá.")
        else:
            rated_items = np.where(train_matrix[user_idx] > 0)[0]
            if len(rated_items) > 0:
                history_list = []
                for item_idx in rated_items:
                    rating = train_matrix[user_idx, item_idx]
                    name = movie_titles.get(int(item_idx) + 1, f"Phim {item_idx + 1}")
                    history_list.append({"title": name, "score": f"Đã chấm: {rating} ⭐"})
                history_list.sort(key=lambda x: float(x['score'].split(': ')[1].split(' ')[0]), reverse=True)
                
                view_mode = st.radio("Chế độ hiển thị", ["Danh sách mở rộng (Có hình ảnh)", "Danh sách rút gọn"], horizontal=True)
                
                if view_mode == "Danh sách mở rộng (Có hình ảnh)":
                    render_movie_grid(history_list, tmdb_key, cols_count=5)
                else:
                    df_history = pd.DataFrame(history_list)
                    df_history.columns = ['Tên Phim', 'Điểm Đánh Giá']
                    # Sắp xếp lại chỉ số rank
                    df_history.index = np.arange(1, len(df_history) + 1)
                    st.table(df_history)
            else:
                st.info("Người dùng này chưa xem/đánh giá bộ phim nào.")
                
    with tab_rating:
        st.subheader("Gửi đánh giá phim")
        movie_options = [f"{id} - {title}" for id, title in movie_titles.items()]
        
        # Thanh tìm kiếm
        search_query = st.text_input("Tìm kiếm phim theo tên...")
        
        if search_query:
            filtered_options = [opt for opt in movie_options if search_query.lower() in opt.lower()]
        else:
            filtered_options = movie_options
            
        with st.form("new_rating_form"):
            selected_movie = st.selectbox("Chọn phim", filtered_options)
            rating = st.slider("Đánh giá (Sao)", 1, 5, 5)
            submit_rating = st.form_submit_button("Gửi Đánh Giá")
            
            if submit_rating:
                if selected_movie:
                    try:
                        movie_id = int(selected_movie.split(" - ")[0])
                        add_movie_rating(current_user, movie_id, rating)
                        st.success("Cảm ơn bạn đã đánh giá! Dữ liệu đã được cập nhật trực tiếp vào SQL Server.")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                else:
                    st.warning("Vui lòng chọn một bộ phim.")
                    
        # Mặc định hiển thị gợi ý
        if not search_query:
            st.markdown("---")
            st.markdown("**Gợi ý phim bạn có thể thích (SVD):**")
            user_recs = get_svd_recommendations(current_user, top_n=5)
            if user_recs:
                render_movie_grid(user_recs, tmdb_key, cols_count=5)

# ==========================================
# TRANG DEVELOPER/ADMIN
# ==========================================
elif page == "Dành Cho Developer":
    # Sử dụng cột để giới hạn độ rộng của tab trên màn hình wide
    _, center_col, _ = st.columns([1, 6, 1])
    
    with center_col:
        st.title("⚙️ Bảng Điều Khiển & Đánh Giá Thuật Toán")
        
        tab1, tab2, tab3 = st.tabs(["📊 Trực Quan Hóa Dữ Liệu", "📈 Đánh Giá Hiệu Năng", "🧪 So Sánh Thuật Toán"])
        
        with tab1:
            st.header("Khám Phá Cơ Chế Các Thuật Toán Gợi Ý")
        
            c_prob1, c_prob2 = st.columns(2)
            with c_prob1:
                total_elements = train_matrix.shape[0] * train_matrix.shape[1]
                nonzero_elements = np.count_nonzero(train_matrix)
                sparsity = (1 - nonzero_elements / total_elements) * 100
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.pie([nonzero_elements, total_elements - nonzero_elements], 
                       labels=['Có đánh giá', 'Trống'], 
                       autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, explode=(0.1, 0))
                ax.set_title(f"Mức Độ Thưa Thớt ({sparsity:.1f}%)", fontsize=10)
                st.pyplot(fig)
                
            with c_prob2:
                fig, ax = plt.subplots(figsize=(4, 3))
                movie_rating_counts = df_raw['item_id'].value_counts().values
                ax.plot(movie_rating_counts, color='green', linewidth=2)
                ax.fill_between(range(len(movie_rating_counts)), movie_rating_counts, color='green', alpha=0.3)
                ax.set_title("Hiện tượng Đuôi Dài (Long-Tail)")
                ax.set_ylabel("Số lượng đánh giá")
                ax.set_xticks([])
                st.pyplot(fig)
                
            st.markdown("---")
            
            st.subheader("Ma trận tương đồng (User Similarity Heatmap)")
            st.markdown("Trực quan hóa độ tương đồng Pearson giữa một số User:")
            sample_users = st.multiselect("Chọn các User ID:", list(range(1, 31)), default=[1, 2, 3, 4, 5])
            if len(sample_users) > 1:
                indices = [u - 1 for u in sample_users]
                sim_sub = user_cf.similarity_matrix[np.ix_(indices, indices)]
                fig_sim, ax_sim = plt.subplots(figsize=(5, 4))
                sns.heatmap(sim_sub, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=sample_users, yticklabels=sample_users, ax=ax_sim, annot_kws={"size": 8})
                ax_sim.set_title("User Similarity Matrix (Pearson)", fontsize=10)
                st.pyplot(fig_sim)

            st.markdown("---")

            st.subheader("Matrix Factorization (SVD): Trực quan hóa Đặc trưng ẩn")
            @st.cache_data
            def get_pca_data(_q_matrix, _movies):
                from sklearn.decomposition import PCA
                q_subset = _q_matrix[_movies]
                return PCA(n_components=2).fit_transform(q_subset)

            top_100_movies = df_raw['item_id'].value_counts().head(100).index.values
            if svd_model is not None and getattr(svd_model, 'Q', None) is not None:
                q_pca = get_pca_data(svd_model.Q, top_100_movies)
                
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(q_pca[:, 0], q_pca[:, 1], alpha=0.6, c=q_pca[:, 0], cmap='viridis', s=60, edgecolors='w')
                
                np.random.seed(42)
                for idx in np.random.choice(100, 15, replace=False):
                    movie_name = movie_titles.get(int(top_100_movies[idx]), f"Phim {top_100_movies[idx]}")
                    short_name = movie_name[:20] + "..." if len(movie_name) > 20 else movie_name
                    ax.annotate(short_name, (q_pca[idx, 0], q_pca[idx, 1]), xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                ax.set_title("Không Gian Đặc Trưng Ẩn 2D Của Top 100 Phim Phổ Biến", fontsize=10)
                st.pyplot(fig)
                
        with tab2:
            st.header("So Sánh Độ Lệch Sai Số (MAE/RMSE)")
            
            if svd_model and len(svd_model.history.get('epoch', [])) > 0:
                fig, ax = plt.subplots(figsize=(5, 3))
                epochs = svd_model.history['epoch']
                ax.plot(epochs, svd_model.history['train_rmse'], label='Train RMSE', marker='o', color='blue', markersize=4)
                ax.plot(epochs, svd_model.history['test_rmse'], label='Test RMSE', marker='s', color='red', markersize=4)
                ax.set_xlabel('Epochs')
                ax.set_ylabel('RMSE')
                ax.set_title("SVD Training Loss")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Chưa có dữ liệu lịch sử huấn luyện SVD.")
                    
            st.markdown("---")
            if st.button("📊 Chạy So Sánh Thuật Toán"):
                with st.spinner("Đang tính toán đánh giá..."):
                    # Sử dụng Global Baseline làm mốc so sánh
                    baseline_model = item_cf.baseline_predictor
                    
                    eval_data = {
                        'Thuật toán': ['Global Baseline', 'User-Based (Biased)', 
                                       'Item-Based (Biased)', 'SVD (Matrix Factorization)'],
                        'MAE': [
                            compute_mae(test_matrix, baseline_model) if baseline_model else 0.0,
                            compute_mae(test_matrix, user_cf),
                            compute_mae(test_matrix, item_cf),
                            compute_mae(test_matrix, svd_model)
                        ],
                        'RMSE': [
                            compute_rmse(test_matrix, baseline_model) if baseline_model else 0.0,
                            compute_rmse(test_matrix, user_cf),
                            compute_rmse(test_matrix, item_cf),
                            compute_rmse(test_matrix, svd_model)
                        ]
                    }
                    eval_df = pd.DataFrame(eval_data)
                    eval_df.set_index('Thuật toán', inplace=True)
                    st.markdown("**Bảng So Sánh Sai Số Dự Đoán trên Test Set**")
                    st.dataframe(eval_df.style.highlight_min(axis=0, color='lightgreen').format("{:.4f}"))

        with tab3:
            st.header("So Sánh Gợi Ý Trực Tiếp")
            st.markdown("So sánh song song các thuật toán gợi ý.")
            
            c_input1, c_input2 = st.columns(2)
            with c_input1:
                test_user = st.number_input("Chọn User ID để so sánh:", min_value=1, max_value=st.session_state.get('max_user_id', train_matrix.shape[0]), value=1, key='test_user')
            with c_input2:
                top_k = st.slider("Số lượng bộ phim muốn gợi ý:", min_value=1, max_value=20, value=5, key='top_k_slider')
            
            if "show_comparison_details" not in st.session_state:
                st.session_state.show_comparison_details = False
                
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Gợi ý phim", use_container_width=True):
                    st.session_state.show_comparison_details = False # Reset chi tiết
                    st.session_state.run_basic_comparison = True
            with col_btn2:
                if st.session_state.get("run_basic_comparison", False):
                    if st.button("Xem chi tiết so sánh", use_container_width=True):
                        st.session_state.show_comparison_details = True
            
            # Logic chạy dự đoán
            if st.session_state.get("run_basic_comparison", False):
                user_idx = test_user - 1
                if user_idx >= train_matrix.shape[0]:
                    st.warning("User mới, chưa có trong ma trận huấn luyện, vui lòng thử ID khác.")
                else:
                    unviewed_items = np.where(train_matrix[user_idx] == 0)[0]
                    if len(unviewed_items) == 0:
                        st.warning("User này đã xem tất cả các phim!")
                    else:
                        with st.spinner("Đang chạy dự đoán..."):
                            preds_user = user_cf.predict_batch(user_idx, unviewed_items)
                            preds_item = item_cf.predict_batch(user_idx, unviewed_items)
                            preds_svd = svd_model.predict_batch(user_idx, unviewed_items)
                            
                            top_user_idx = unviewed_items[np.argsort(preds_user)[-top_k:][::-1]]
                            top_item_idx = unviewed_items[np.argsort(preds_item)[-top_k:][::-1]]
                            top_svd_idx = unviewed_items[np.argsort(preds_svd)[-top_k:][::-1]]
                            
                            if not st.session_state.get("show_comparison_details", False):
                                # Hiển thị SVD mặc định
                                st.subheader(f"Kết quả gợi ý từ SVD ({top_k} phim):")
                                user_recs = []
                                top_scores = preds_svd[np.argsort(preds_svd)[-top_k:][::-1]]
                                for rank, (idx, score) in enumerate(zip(top_svd_idx, top_scores), 1):
                                    name = movie_titles.get(int(idx) + 1, f"Phim {idx+1}")
                                    user_recs.append({"title": name, "score": f"{score:.2f} ⭐", "rank": rank})
                                render_movie_grid(user_recs, tmdb_key, cols_count=5)
                            else:
                                st.markdown("---")
                                st.subheader("Chi Tiết So Sánh 3 Thuật Toán")
                                c1, c2, c3 = st.columns(3)
                                
                                with c1:
                                    st.markdown("### 👤 User-Based CF")
                                    user_cf_mode = st.selectbox("Phương pháp:", ["biased_baseline", "means"], key="user_cf_mode")
                                    
                                    # Tạo instance tạm để chạy mode khác
                                    temp_user_cf = UserBasedCollaborativeFiltering(k_neighbors=user_cf.k_neighbors, prediction_mode=user_cf_mode)
                                    temp_user_cf.train_matrix = user_cf.train_matrix
                                    temp_user_cf.similarity_matrix = user_cf.similarity_matrix
                                    temp_user_cf.user_means = user_cf.user_means
                                    if user_cf_mode == 'biased_baseline':
                                        temp_user_cf.baseline_predictor = user_cf.baseline_predictor
                                        
                                    t_preds_user = temp_user_cf.predict_batch(user_idx, unviewed_items)
                                    t_top_user_idx = unviewed_items[np.argsort(t_preds_user)[-top_k:][::-1]]
                                    
                                    for rank, idx in enumerate(t_top_user_idx, 1):
                                        name = movie_titles.get(int(idx) + 1, f"Phim {idx+1}")
                                        st.markdown(f"**{rank}.** {name}")
                                        
                                with c2:
                                    st.markdown("### 📦 Item-Based CF")
                                    item_cf_mode = st.selectbox("Phương pháp:", ["biased_baseline", "pure"], key="item_cf_mode")
                                    
                                    temp_item_cf = ItemBasedCollaborativeFiltering(k_neighbors=item_cf.k_neighbors, prediction_mode=item_cf_mode)
                                    temp_item_cf.train_matrix = item_cf.train_matrix
                                    temp_item_cf.similarity_matrix = item_cf.similarity_matrix
                                    if item_cf_mode == 'biased_baseline':
                                        temp_item_cf.baseline_predictor = item_cf.baseline_predictor
                                        
                                    t_preds_item = temp_item_cf.predict_batch(user_idx, unviewed_items)
                                    t_top_item_idx = unviewed_items[np.argsort(t_preds_item)[-top_k:][::-1]]
                                    
                                    for rank, idx in enumerate(t_top_item_idx, 1):
                                        name = movie_titles.get(int(idx) + 1, f"Phim {idx+1}")
                                        st.markdown(f"**{rank}.** {name}")
                                        
                                with c3:
                                    st.markdown("### 🧠 SVD")
                                    st.markdown("*Matrix Factorization*")
                                    for rank, idx in enumerate(top_svd_idx, 1):
                                        name = movie_titles.get(int(idx) + 1, f"Phim {idx+1}")
                                        st.markdown(f"**{rank}.** {name}")