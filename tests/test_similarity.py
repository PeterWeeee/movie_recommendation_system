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
    # Cosine(User0, User1) = (5*4) / (sqrt(34) * 4) = 5 / sqrt(34)
    expected_sim_0_1 = 5 / np.sqrt(34)
    assert np.isclose(sim[0, 1], expected_sim_0_1)
    assert np.isclose(sim[1, 2], 0.0) # Không có phim chung

def test_compute_pearson_similarity():
    matrix = np.array([
        [5, 4, 0],
        [4, 5, 0],
        [1, 1, 0]
    ], dtype=float)
    
    # User 0 đã đánh giá 2 bộ phim. Trọng số = 2/50 = 0.04
    sim = compute_pearson_similarity(matrix, gamma=50)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 0], 1.0 * (2/50))

def test_compute_adjusted_cosine_similarity():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    # matrix shape (3, 3) -> adjusted cosine computes similarity between items -> (3, 3)
    # Item 0 được đánh giá bởi User 0 và User 1 (2 người). Trọng số = 2/50 = 0.04
    sim = compute_adjusted_cosine_similarity(matrix, gamma=50)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 0], 1.0 * (2/50))
