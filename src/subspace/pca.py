"""
Classical PCA for Dimensionality Reduction
Projects skeleton sequences from 60D to quantum-friendly dimension d_q.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class SkeletonPCA:
    """
    PCA-based dimensionality reduction for skeleton sequences.
    
    Reduces from 60D (20 joints × 3 coords) to d_q (quantum dimension).
    This makes quantum circuits much cheaper to run.
    """
    
    def __init__(self, n_components: int = 8):
        """
        Initialize PCA projector.
        
        Args:
            n_components: Target dimension (d_q)
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        
    def fit(self, sequences: List[np.ndarray]):
        """
        Fit PCA on training sequences.
        
        Args:
            sequences: List of sequences, each of shape (T_i, 60)
        """
        # Stack all frames from all sequences
        all_frames = np.vstack(sequences)
        
        logger.info(f"Fitting PCA on {len(all_frames)} frames from {len(sequences)} sequences")
        logger.info(f"Original dimension: {all_frames.shape[1]}, Target: {self.n_components}")
        
        # Fit PCA
        self.pca.fit(all_frames)
        self.fitted = True
        
        # Log variance explained
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"Explained variance: {explained_var*100:.2f}%")
        logger.info(f"First {min(5, self.n_components)} components: "
                   f"{self.pca.explained_variance_ratio_[:5]}")
        
    def transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Project a single sequence to low-dimensional space.
        
        Args:
            sequence: Original sequence (T, 60)
            
        Returns:
            Projected sequence (T, d_q)
        """
        if not self.fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.transform(sequence)
    
    def transform_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Project multiple sequences.
        
        Args:
            sequences: List of original sequences
            
        Returns:
            List of projected sequences
        """
        return [self.transform_sequence(seq) for seq in sequences]
    
    def inverse_transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Reconstruct sequence from low-dimensional representation.
        
        Args:
            sequence: Projected sequence (T, d_q)
            
        Returns:
            Reconstructed sequence (T, 60)
        """
        if not self.fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.inverse_transform(sequence)
    
    def reconstruction_error(self, sequence: np.ndarray) -> float:
        """
        Compute reconstruction error for a sequence.
        
        Args:
            sequence: Original sequence (T, 60)
            
        Returns:
            Mean squared reconstruction error
        """
        projected = self.transform_sequence(sequence)
        reconstructed = self.inverse_transform_sequence(projected)
        return np.mean((sequence - reconstructed) ** 2)
    
    def get_components(self) -> np.ndarray:
        """
        Get PCA components (principal directions).
        
        Returns:
            Components matrix (d_q, 60)
        """
        if not self.fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.components_


def project_dataset(
    train_sequences: List[np.ndarray],
    test_sequences: List[np.ndarray],
    n_components: int = 8
) -> Tuple[List[np.ndarray], List[np.ndarray], SkeletonPCA]:
    """
    Convenience function to project entire dataset.
    
    Args:
        train_sequences: Training sequences (60D)
        test_sequences: Test sequences (60D)
        n_components: Target dimension
        
    Returns:
        (projected_train, projected_test, pca_model)
    """
    # Fit PCA on training data
    pca = SkeletonPCA(n_components=n_components)
    pca.fit(train_sequences)
    
    # Project both train and test
    projected_train = pca.transform_sequences(train_sequences)
    projected_test = pca.transform_sequences(test_sequences)
    
    logger.info(f"\nProjected dataset to {n_components}D:")
    logger.info(f"  Train: {len(projected_train)} sequences, shape {projected_train[0].shape}")
    logger.info(f"  Test: {len(projected_test)} sequences, shape {projected_test[0].shape}")
    
    return projected_train, projected_test, pca


if __name__ == "__main__":
    # Test PCA
    np.random.seed(42)
    
    # Create synthetic data
    n_sequences = 100
    seq_length = 50
    original_dim = 60
    
    sequences = [np.random.randn(seq_length, original_dim) for _ in range(n_sequences)]
    
    # Test projection
    pca = SkeletonPCA(n_components=8)
    pca.fit(sequences[:80])  # Fit on subset
    
    # Transform
    projected = pca.transform_sequences(sequences)
    
    print(f"Original shape: {sequences[0].shape}")
    print(f"Projected shape: {projected[0].shape}")
    
    # Test reconstruction
    error = pca.reconstruction_error(sequences[0])
    print(f"Reconstruction error: {error:.6f}")
