import numpy as np
import pytest
from src.recommender import BiasedPredictor, UserBasedCollaborativeFiltering, ItemBasedCollaborativeFiltering, MatrixFactorizationSVD
def test_biased_predictor():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    predictor = BiasedPredictor()
    predictor.fit(matrix)
    
    # mu = (5+3+4+2+5)/5 = 19/5 = 3.8
    assert np.isclose(predictor.mu, 3.8)
    
    pred = predictor.predict_rating(0, 2)
    assert 1.0 <= pred <= 5.0

def test_user_based_cf_means():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    sim_matrix = np.array([
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.0],
        [0.1, 0.0, 1.0]
    ])
    
    model = UserBasedCollaborativeFiltering(k_neighbors=2, prediction_mode='means')
    model.fit(matrix, sim_matrix)
    
    pred = model.predict_rating(1, 1) # User 1 predict item 1
    assert 1.0 <= pred <= 5.0

def test_svd():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    model = MatrixFactorizationSVD(num_factors=2, epochs=5)
    model.fit(matrix, matrix) # test=train
    
    pred = model.predict_rating(1, 1)
    assert 1.0 <= pred <= 5.0
    
    assert len(model.history['epoch']) == 5
    assert model.history['train_rmse'][-1] >= 0.0

def test_item_based_cf():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    sim_matrix = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.1],
        [0.2, 0.1, 1.0]
    ])
    
    model = ItemBasedCollaborativeFiltering(k_neighbors=2)
    model.fit(matrix, sim_matrix)
    
    pred = model.predict_rating(0, 2)
    assert 1.0 <= pred <= 5.0
    
def test_predict_batch():
    matrix = np.array([
        [5, 3, 0],
        [4, 0, 0],
        [0, 2, 5]
    ], dtype=float)
    
    model = MatrixFactorizationSVD(num_factors=2, epochs=5)
    model.fit(matrix, matrix)
    
    unviewed = np.array([1, 2])
    preds = model.predict_batch(1, unviewed)
    
    assert len(preds) == 2
    assert all(1.0 <= p <= 5.0 for p in preds)
