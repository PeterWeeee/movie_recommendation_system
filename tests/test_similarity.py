import numpy as np
import pytest
from src.similarity import compute_cosine_similarity, compute_pearson_similarity, compute_adjusted_cosine_similarity

def test_compute_cosine_similarity():
    matrix = np.array([
        [5, 0, 3],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    sim = compute_cosine_similarity(matrix)
    assert sim.shape == (3, 3)
    # Co-rated Cosine(User0, User1): Phim chung duy nhất là item 0 (User 0 chấm 5, User 1 chấm 4). 
    # Tử số: 5*4 = 20. Mẫu số: sqrt(5^2) * sqrt(4^2) = 5 * 4 = 20. Kết quả = 1.0
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[0, 1], 1.0)
    assert np.isclose(sim[1, 2], 0.0) # Không có phim chung

def test_compute_pearson_similarity():
    matrix = np.array([
        [5, 4, 0],
        [4, 5, 0],
        [1, 1, 0]
    ], dtype=float)
    
    sim = compute_pearson_similarity(matrix)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 0], 1.0)

def test_compute_adjusted_cosine_similarity():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    # matrix shape (3, 3) -> adjusted cosine computes similarity between items -> (3, 3)
    sim = compute_adjusted_cosine_similarity(matrix)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 0], 1.0)
