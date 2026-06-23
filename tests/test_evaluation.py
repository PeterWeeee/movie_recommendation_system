import numpy as np
import pytest
from src.evaluation import (
    compute_mae,
    compute_rmse,
    compute_precision_recall_at_k,
    compute_f1_at_k,
    compute_ndcg_at_k
)

class MockModel:
    def predict_batch(self, user_idx, item_indices):
        # Dummy prediction: returns 4.0 for even items, 3.0 for odd items
        preds = np.zeros(len(item_indices))
        for idx, item in enumerate(item_indices):
            preds[idx] = 4.0 if item % 2 == 0 else 3.0
        return preds

def test_mae_rmse():
    test_matrix = np.array([
        [5, 0, 3],
        [0, 4, 0],
        [0, 0, 5]
    ], dtype=float)
    
    model = MockModel()
    
    # Nonzero positions are (0,0)->5, (0,2)->3, (1,1)->4, (2,2)->5
    # Predictions:
    # (0,0): item 0 (even) -> 4.0. Diff = |5 - 4| = 1.0
    # (0,2): item 2 (even) -> 4.0. Diff = |3 - 4| = 1.0
    # (1,1): item 1 (odd) -> 3.0. Diff = |4 - 3| = 1.0
    # (2,2): item 2 (even) -> 4.0. Diff = |5 - 4| = 1.0
    # MAE = (1 + 1 + 1 + 1) / 4 = 1.0
    # RMSE = sqrt((1^2 + 1^2 + 1^2 + 1^2) / 4) = 1.0
    
    mae = compute_mae(test_matrix, model)
    rmse = compute_rmse(test_matrix, model)
    
    assert np.isclose(mae, 1.0)
    assert np.isclose(rmse, 1.0)

def test_precision_recall_f1():
    train_matrix = np.array([
        [0, 5, 0, 0],
        [0, 0, 0, 0]
    ], dtype=float)
    
    test_matrix = np.array([
        [4, 0, 5, 0],
        [0, 0, 0, 0]
    ], dtype=float)
    
    model = MockModel()
    
    # User 0:
    # Test items >= 3.5: item 0 (val 4), item 2 (val 5) -> [0, 2]
    # Unviewed items in train: item 0, item 2, item 3 -> [0, 2, 3]
    # Predictions for [0, 2, 3]:
    # item 0 (even) -> 4.0
    # item 2 (even) -> 4.0
    # item 3 (odd) -> 3.0
    # argargsort(preds) on [0, 2, 3]:
    # preds: [4.0, 4.0, 3.0]
    # Sorted indices: [3, 0, 2] (ascending) or [2, 0, 3] / [0, 2, 3] (depending on stable sort)
    # top k=2 indices will be [0, 2]
    # Hits: [0, 2] intersect [0, 2] = 2 hits
    # Precision@2 = 2 / 2 = 1.0
    # Recall@2 = 2 / 2 = 1.0
    # F1@2 = 1.0
    
    precision, recall = compute_precision_recall_at_k(train_matrix, test_matrix, model, k=2, threshold=3.5)
    f1 = compute_f1_at_k(precision, recall)
    
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 1.0)
    assert np.isclose(f1, 1.0)

def test_ndcg():
    train_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=float)
    
    test_matrix = np.array([
        [5, 3, 0],
        [0, 0, 0]
    ], dtype=float)
    
    model = MockModel()
    ndcg = compute_ndcg_at_k(train_matrix, test_matrix, model, k=2)
    assert ndcg >= 0.0
