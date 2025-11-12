"""
Multi-scale quantum similarity measurement.

Tests quantum similarity at multiple qubit granularities and combines results
for more robust correlation with classical DTW.
"""
import numpy as np
from typing import List, Tuple
import logging

from .amplitude_encoding import amplitude_encode
from .swap_test import quantum_swap_test

logger = logging.getLogger(__name__)


def multiscale_quantum_similarity(
    vector1: np.ndarray,
    vector2: np.ndarray,
    qubit_scales: List[int] = [3, 4, 5],
    shots: int = 1024,
    weights: List[float] = None
) -> Tuple[float, dict]:
    """
    Compute quantum similarity at multiple scales and combine.
    
    Args:
        vector1: First vector (already PCA-reduced)
        vector2: Second vector (already PCA-reduced)
        qubit_scales: List of qubit counts to test [3, 4, 5]
        shots: Number of shots per swap test
        weights: Weighting for each scale (default: equal weights)
        
    Returns:
        (combined_cost, details_dict) where details contains per-scale costs
    """
    if weights is None:
        weights = [1.0 / len(qubit_scales)] * len(qubit_scales)
    
    if len(weights) != len(qubit_scales):
        raise ValueError("weights must match qubit_scales length")
    
    scale_costs = []
    scale_details = {}
    
    for n_qubits, weight in zip(qubit_scales, weights):
        # Determine dimensions needed for this scale
        n_dims = 2 ** n_qubits
        
        # Resize vectors if needed
        if len(vector1) < n_dims:
            # Pad with zeros
            v1 = np.pad(vector1, (0, n_dims - len(vector1)))
            v2 = np.pad(vector2, (0, n_dims - len(vector2)))
        elif len(vector1) > n_dims:
            # Truncate
            v1 = vector1[:n_dims]
            v2 = vector2[:n_dims]
        else:
            v1 = vector1.copy()
            v2 = vector2.copy()
        
        # Encode to quantum states
        state1 = amplitude_encode(v1, n_qubits=n_qubits)
        state2 = amplitude_encode(v2, n_qubits=n_qubits)
        
        # Compute swap test similarity
        similarity = quantum_swap_test(state1, state2, shots=shots)
        
        # Convert to cost (1 - similarity gives distance-like measure)
        cost = 1.0 - similarity
        
        scale_costs.append(cost * weight)
        scale_details[f'{n_qubits}_qubits'] = {
            'cost': cost,
            'similarity': similarity,
            'weight': weight,
            'weighted_cost': cost * weight
        }
        
        logger.debug(f"Scale {n_qubits} qubits: similarity={similarity:.4f}, "
                    f"cost={cost:.4f}, weighted={cost*weight:.4f}")
    
    # Combine weighted costs
    combined_cost = sum(scale_costs)
    
    return combined_cost, scale_details


def multiscale_quantum_cost_matrix(
    sequences: List[np.ndarray],
    qubit_scales: List[int] = [3, 4, 5],
    shots: int = 1024,
    weights: List[float] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Compute pairwise multi-scale quantum similarity for all sequence pairs.
    
    Args:
        sequences: List of PCA-reduced sequence features
        qubit_scales: List of qubit counts to test
        shots: Number of shots per swap test
        weights: Weighting for each scale
        
    Returns:
        (cost_matrix, details_list) where cost_matrix is NxN and details 
        contains per-pair scale information
    """
    n = len(sequences)
    cost_matrix = np.zeros((n, n))
    details_list = []
    
    for i in range(n):
        for j in range(i+1, n):
            cost, details = multiscale_quantum_similarity(
                sequences[i],
                sequences[j],
                qubit_scales=qubit_scales,
                shots=shots,
                weights=weights
            )
            
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost  # Symmetric
            
            details_list.append({
                'pair': (i, j),
                'cost': cost,
                'scales': details
            })
            
            logger.info(f"Pair ({i},{j}): combined_cost={cost:.4f}")
    
    return cost_matrix, details_list
