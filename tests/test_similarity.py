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
    # Cosine(User0, User1) = (5*4) / (sqrt(34) * sqrt(16)) = 20 / (5.83 * 4) = 0.857
    assert np.isclose(sim[0, 0], 1.0)
    assert sim[0, 1] > 0.8
    assert np.isclose(sim[1, 2], 0.0) # orthogonal

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
