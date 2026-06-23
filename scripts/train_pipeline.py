import os
import sys
import pickle

# Thêm thư mục gốc vào sys.path để có thể import từ src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.data_loader import get_or_create_processed_matrices
from src.similarity import compute_pearson_similarity, compute_cosine_similarity, compute_adjusted_cosine_similarity
from src.recommender import UserBasedCollaborativeFiltering, ItemBasedCollaborativeFiltering, MatrixFactorizationSVD
from src.content_based import ContentBasedRecommender, load_item_metadata

def main():
    print("Starting Training Pipeline...")
    data_path = os.path.join(base_dir, 'data', 'raw', 'u.data')
    item_path = os.path.join(base_dir, 'data', 'raw', 'u.item')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("1. Loading and processing data...")
    train_matrix, test_matrix = get_or_create_processed_matrices(data_path, processed_dir)
    
    print("2. Computing Similarity Matrices...")
    print(" - Computing User-User Cosine Similarity...")
    user_cosine_sim = compute_cosine_similarity(train_matrix)
    print(" - Computing User-User Pearson Similarity...")
    user_pearson_sim = compute_pearson_similarity(train_matrix)
    print(" - Computing Item-Item Cosine Similarity...")
    item_cosine_sim = compute_cosine_similarity(train_matrix.T)  # dùng cột (items) làm vector
    print(" - Computing Item-Item Adjusted Cosine Similarity...")
    item_adj_cosine_sim = compute_adjusted_cosine_similarity(train_matrix)
    
    print("3. Initializing and Saving Memory-Based Models...")
    user_cf = UserBasedCollaborativeFiltering(k_neighbors=40)
    user_cf.fit_both(train_matrix, user_cosine_sim, user_pearson_sim)
    with open(os.path.join(models_dir, 'user_cf.pkl'), 'wb') as f:
        pickle.dump(user_cf, f)
        
    item_cf = ItemBasedCollaborativeFiltering(k_neighbors=40)
    item_cf.fit_both(train_matrix, item_cosine_sim, item_adj_cosine_sim)
    with open(os.path.join(models_dir, 'item_cf.pkl'), 'wb') as f:
        pickle.dump(item_cf, f)
        
    print("4. Training Matrix Factorization (SVD)...")
    svd_model = MatrixFactorizationSVD(num_factors=20, lr=0.005, reg=0.02, epochs=20)
    svd_model.fit(train_matrix, test_matrix)
    svd_model.save_model(os.path.join(models_dir, 'svd_weights.pkl'))
    
    print("5. Training Content-Based Model...")
    df_item = load_item_metadata(item_path)
    content_model = ContentBasedRecommender()
    content_model.fit(df_item)
    with open(os.path.join(models_dir, 'content_based.pkl'), 'wb') as f:
        pickle.dump(content_model, f)
        
    print("Pipeline completed successfully! All models are saved in 'models/' directory.")

if __name__ == "__main__":
    main()
