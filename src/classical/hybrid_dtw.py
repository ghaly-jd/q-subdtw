"""
Hybrid Quantum-Classical DTW Distance.

Combines classical DTW alignment with quantum similarity measures
for improved skeleton sequence matching.
"""
import numpy as np
from typing import Optional, Tuple
import logging

from .dtw_distance import DTWDistance
from ..quantum.amplitude_encoding import amplitude_encode
from ..quantum.swap_test import quantum_swap_test
from ..quantum.quantum_multiscale import multiscale_quantum_similarity

logger = logging.getLogger(__name__)


class HybridDTW:
    """
    Hybrid DTW combining classical and quantum costs.
    
    Uses classical DTW for temporal alignment and quantum swap test
    for feature-level similarity, combining with weighted average.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        window: Optional[int] = 10,
        use_multiscale: bool = False,
        qubit_scales: list = None,
        quantum_shots: int = 1024
    ):
        """
        Initialize hybrid DTW.
        
        Args:
            alpha: Weight for classical cost (1-alpha for quantum)
                   alpha=1.0 means pure classical, alpha=0.0 means pure quantum
            window: Sakoe-Chiba band window (None for no constraint)
            use_multiscale: Whether to use multi-scale quantum similarity
            qubit_scales: List of qubit counts for multi-scale [3,4,5]
            quantum_shots: Number of shots for quantum measurements
        """
        self.alpha = alpha
        self.classical_dtw = DTWDistance(window=window)
        self.use_multiscale = use_multiscale
        self.qubit_scales = qubit_scales if qubit_scales else [3, 4, 5]
        self.quantum_shots = quantum_shots
        
        logger.info(f"HybridDTW initialized: alpha={alpha}, window={window}, "
                   f"multiscale={use_multiscale}")
    
    def _quantum_similarity(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray
    ) -> float:
        """
        Compute quantum similarity between sequences.
        
        Args:
            seq1, seq2: PCA-reduced sequence features (T x D)
            
        Returns:
            Quantum cost (distance-like measure)
        """
        # Average features over time to get single vectors
        vec1 = np.mean(seq1, axis=0)
        vec2 = np.mean(seq2, axis=0)
        
        if self.use_multiscale:
            # Multi-scale quantum similarity
            cost, _ = multiscale_quantum_similarity(
                vec1, vec2,
                qubit_scales=self.qubit_scales,
                shots=self.quantum_shots
            )
            return cost
        else:
            # Single-scale quantum similarity (3 qubits)
            n_qubits = 3
            n_dims = 2 ** n_qubits
            
            # Resize vectors
            if len(vec1) < n_dims:
                vec1 = np.pad(vec1, (0, n_dims - len(vec1)))
                vec2 = np.pad(vec2, (0, n_dims - len(vec2)))
            elif len(vec1) > n_dims:
                vec1 = vec1[:n_dims]
                vec2 = vec2[:n_dims]
            
            # Encode and measure
            state1 = amplitude_encode(vec1, n_qubits=n_qubits)
            state2 = amplitude_encode(vec2, n_qubits=n_qubits)
            similarity = quantum_swap_test(state1, state2, shots=self.quantum_shots)
            
            # Convert to cost
            return 1.0 - similarity
    
    def distance(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        return_components: bool = False
    ) -> float:
        """
        Compute hybrid distance between sequences.
        
        Args:
            seq1, seq2: PCA-reduced sequences (T x D)
            return_components: If True, return (hybrid, classical, quantum)
            
        Returns:
            Hybrid distance, or tuple if return_components=True
        """
        # Classical DTW cost
        classical_cost = self.classical_dtw.distance(seq1, seq2)
        
        # Quantum cost
        quantum_cost = self._quantum_similarity(seq1, seq2)
        
        # Combine with weighted average
        # Note: Need to normalize since classical and quantum have different scales
        # We'll use relative scaling based on empirical observations
        # From experiments: classical ~200-600, quantum ~5-45
        # Scale factor: ~20x difference
        scale_factor = 20.0
        normalized_quantum = quantum_cost * scale_factor
        
        hybrid_cost = self.alpha * classical_cost + (1 - self.alpha) * normalized_quantum
        
        logger.debug(f"Costs - Classical: {classical_cost:.2f}, "
                    f"Quantum: {quantum_cost:.2f}, "
                    f"Normalized Quantum: {normalized_quantum:.2f}, "
                    f"Hybrid: {hybrid_cost:.2f}")
        
        if return_components:
            return hybrid_cost, classical_cost, quantum_cost
        return hybrid_cost
    
    def distance_matrix(
        self,
        sequences: list,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pairwise distance matrix for sequences.
        
        Args:
            sequences: List of PCA-reduced sequences
            verbose: Whether to print progress
            
        Returns:
            (hybrid_matrix, classical_matrix, quantum_matrix)
        """
        n = len(sequences)
        hybrid_matrix = np.zeros((n, n))
        classical_matrix = np.zeros((n, n))
        quantum_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                h, c, q = self.distance(
                    sequences[i],
                    sequences[j],
                    return_components=True
                )
                
                hybrid_matrix[i, j] = hybrid_matrix[j, i] = h
                classical_matrix[i, j] = classical_matrix[j, i] = c
                quantum_matrix[i, j] = quantum_matrix[j, i] = q
                
                if verbose and (i*n + j) % 10 == 0:
                    logger.info(f"Computed pair ({i},{j}): h={h:.2f}, c={c:.2f}, q={q:.2f}")
        
        return hybrid_matrix, classical_matrix, quantum_matrix


class HybridClassifier:
    """
    1-NN classifier using hybrid DTW distance.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        window: Optional[int] = 10,
        use_multiscale: bool = False
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            alpha: Weight for classical cost
            window: DTW window constraint
            use_multiscale: Use multi-scale quantum similarity
        """
        self.hybrid_dtw = HybridDTW(
            alpha=alpha,
            window=window,
            use_multiscale=use_multiscale
        )
        self.train_sequences = []
        self.train_labels = []
    
    def fit(self, sequences: list, labels: list):
        """Store training data."""
        self.train_sequences = sequences
        self.train_labels = labels
        logger.info(f"Fitted with {len(sequences)} training samples")
    
    def predict(self, sequence: np.ndarray) -> int:
        """
        Predict label for a sequence using 1-NN.
        
        Args:
            sequence: PCA-reduced test sequence
            
        Returns:
            Predicted label
        """
        min_dist = float('inf')
        pred_label = None
        
        for train_seq, train_label in zip(self.train_sequences, self.train_labels):
            dist = self.hybrid_dtw.distance(sequence, train_seq)
            if dist < min_dist:
                min_dist = dist
                pred_label = train_label
        
        return pred_label
    
    def evaluate(self, sequences: list, labels: list) -> dict:
        """
        Evaluate on test set.
        
        Args:
            sequences: Test sequences
            labels: True labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        correct = 0
        predictions = []
        
        for seq, label in zip(sequences, labels):
            pred = self.predict(seq)
            predictions.append(pred)
            if pred == label:
                correct += 1
        
        accuracy = correct / len(labels)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(labels),
            'predictions': predictions
        }
