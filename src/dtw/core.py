"""
Classical Dynamic Time Warping (DTW) Implementation
Core DTW algorithm with Sakoe-Chiba band constraint.
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        x: First vector (d,)
        y: Second vector (d,)
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(x - y)


def dtw_distance(
    query: np.ndarray,
    candidate: np.ndarray,
    window: Optional[int] = None,
    distance_fn: Callable = euclidean_distance,
    return_path: bool = False
) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Compute DTW distance between two sequences.
    
    Uses dynamic programming with optional Sakoe-Chiba band constraint.
    
    Cost recurrence:
        D[i,j] = δ(q_i, c_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    
    where δ is the local distance function (default: Euclidean).
    
    Args:
        query: Query sequence (T_q, d)
        candidate: Candidate sequence (T_c, d)
        window: Sakoe-Chiba band width (if None, no constraint)
        distance_fn: Local distance function
        return_path: Whether to return the optimal path
        
    Returns:
        (dtw_distance, path) where path is list of (i, j) tuples if requested
    """
    T_q, d_q = query.shape
    T_c, d_c = candidate.shape
    
    if d_q != d_c:
        raise ValueError(f"Dimension mismatch: query={d_q}, candidate={d_c}")
    
    # Initialize cost matrix with infinity
    D = np.full((T_q + 1, T_c + 1), np.inf)
    D[0, 0] = 0.0
    
    # Compute DTW cost matrix
    for i in range(1, T_q + 1):
        # Apply Sakoe-Chiba band constraint
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(T_c + 1, i + window + 1)
        else:
            j_start = 1
            j_end = T_c + 1
        
        for j in range(j_start, j_end):
            # Local cost
            cost = distance_fn(query[i-1], candidate[j-1])
            
            # Recurrence
            D[i, j] = cost + min(
                D[i-1, j],      # insertion
                D[i, j-1],      # deletion
                D[i-1, j-1]     # match
            )
    
    dtw_dist = D[T_q, T_c]
    
    if not return_path:
        return dtw_dist, None
    
    # Backtrack to find optimal path
    path = _backtrack_path(D, T_q, T_c)
    
    return dtw_dist, path


def _backtrack_path(D: np.ndarray, i: int, j: int) -> List[Tuple[int, int]]:
    """
    Backtrack through cost matrix to find optimal path.
    
    Args:
        D: Cost matrix
        i: End row (query length)
        j: End column (candidate length)
        
    Returns:
        List of (i, j) coordinates forming the path (reversed, from end to start)
    """
    path = [(i, j)]
    
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # Choose direction with minimum cost
            candidates = [
                (D[i-1, j-1], i-1, j-1),  # diagonal
                (D[i-1, j], i-1, j),      # up
                (D[i, j-1], i, j-1)       # left
            ]
            _, i, j = min(candidates)
        
        path.append((i, j))
    
    return path[::-1]  # Reverse to get start-to-end order


def dtw_cost_matrix(
    query: np.ndarray,
    candidate: np.ndarray,
    window: Optional[int] = None,
    distance_fn: Callable = euclidean_distance
) -> np.ndarray:
    """
    Compute full DTW cost matrix (for visualization or quantum refinement).
    
    Args:
        query: Query sequence (T_q, d)
        candidate: Candidate sequence (T_c, d)
        window: Sakoe-Chiba band width
        distance_fn: Local distance function
        
    Returns:
        Cost matrix D of shape (T_q+1, T_c+1)
    """
    T_q, d_q = query.shape
    T_c, d_c = candidate.shape
    
    if d_q != d_c:
        raise ValueError(f"Dimension mismatch: query={d_q}, candidate={d_c}")
    
    # Initialize cost matrix
    D = np.full((T_q + 1, T_c + 1), np.inf)
    D[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(1, T_q + 1):
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(T_c + 1, i + window + 1)
        else:
            j_start = 1
            j_end = T_c + 1
        
        for j in range(j_start, j_end):
            cost = distance_fn(query[i-1], candidate[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    return D


class DTWClassifier:
    """
    1-Nearest Neighbor classifier using DTW distance.
    """
    
    def __init__(self, window: Optional[int] = None, distance_fn: Callable = euclidean_distance):
        """
        Initialize DTW classifier.
        
        Args:
            window: Sakoe-Chiba band width
            distance_fn: Local distance function
        """
        self.window = window
        self.distance_fn = distance_fn
        self.train_sequences = None
        self.train_labels = None
        
    def fit(self, train_sequences: List[np.ndarray], train_labels: List[int]):
        """
        Store training data.
        
        Args:
            train_sequences: List of training sequences
            train_labels: List of training labels
        """
        self.train_sequences = train_sequences
        self.train_labels = np.array(train_labels)
        logger.info(f"Fitted DTW classifier with {len(train_sequences)} training sequences")
    
    def predict_one(self, query: np.ndarray, return_distance: bool = False) -> Tuple[int, Optional[float]]:
        """
        Predict label for a single query sequence.
        
        Args:
            query: Query sequence (T, d)
            return_distance: Whether to return the minimum distance
            
        Returns:
            (predicted_label, min_distance) if return_distance else (predicted_label, None)
        """
        if self.train_sequences is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Compute DTW distance to all training sequences
        distances = []
        for train_seq in self.train_sequences:
            dist, _ = dtw_distance(query, train_seq, window=self.window, distance_fn=self.distance_fn)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Find nearest neighbor
        min_idx = np.argmin(distances)
        predicted_label = self.train_labels[min_idx]
        min_distance = distances[min_idx]
        
        if return_distance:
            return predicted_label, min_distance
        else:
            return predicted_label, None
    
    def predict(self, test_sequences: List[np.ndarray], verbose: bool = True) -> np.ndarray:
        """
        Predict labels for multiple test sequences.
        
        Args:
            test_sequences: List of test sequences
            verbose: Whether to show progress
            
        Returns:
            Array of predicted labels
        """
        predictions = []
        
        for idx, query in enumerate(test_sequences):
            pred, _ = self.predict_one(query)
            predictions.append(pred)
            
            if verbose and (idx + 1) % 10 == 0:
                logger.info(f"Predicted {idx + 1}/{len(test_sequences)} sequences")
        
        return np.array(predictions)
    
    def evaluate(
        self,
        test_sequences: List[np.ndarray],
        test_labels: List[int],
        verbose: bool = True
    ) -> dict:
        """
        Evaluate classifier on test set.
        
        Args:
            test_sequences: List of test sequences
            test_labels: List of true labels
            verbose: Whether to show progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        start_time = time.time()
        
        predictions = self.predict(test_sequences, verbose=verbose)
        test_labels = np.array(test_labels)
        
        accuracy = np.mean(predictions == test_labels)
        
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / len(test_sequences)
        
        results = {
            "accuracy": accuracy,
            "total_time": elapsed_time,
            "avg_time_per_sample": avg_time_per_sample,
            "num_test": len(test_sequences),
            "num_train": len(self.train_sequences),
        }
        
        if verbose:
            logger.info(f"\nClassical DTW Results:")
            logger.info(f"  Accuracy: {accuracy*100:.2f}%")
            logger.info(f"  Total time: {elapsed_time:.2f}s")
            logger.info(f"  Avg time per sample: {avg_time_per_sample:.3f}s")
        
        return results


if __name__ == "__main__":
    # Test DTW
    np.random.seed(42)
    
    # Create two simple sequences
    t = np.linspace(0, 4*np.pi, 50)
    seq1 = np.column_stack([np.sin(t), np.cos(t)])
    seq2 = np.column_stack([np.sin(t + 0.5), np.cos(t + 0.5)])
    
    # Compute DTW distance
    dist, path = dtw_distance(seq1, seq2, window=10, return_path=True)
    
    print(f"DTW distance: {dist:.4f}")
    print(f"Path length: {len(path)}")
    print(f"First few path points: {path[:5]}")
