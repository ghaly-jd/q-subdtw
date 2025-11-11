"""
Quantum Swap Test for Frame Similarity
Computes fidelity between two quantum states using the swap test circuit.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.compiler import transpile
from typing import Tuple
import logging

from .amplitude_encoding import amplitude_encode, normalize_vector

logger = logging.getLogger(__name__)


def create_swap_test_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Create a swap test circuit for comparing two n-qubit states.
    
    The swap test circuit:
    1. Ancilla qubit in |0⟩
    2. Two registers with states |ψ⟩ and |φ⟩
    3. Apply H on ancilla
    4. Apply controlled-SWAP between the two registers (controlled by ancilla)
    5. Apply H on ancilla
    6. Measure ancilla
    
    If ancilla measures |0⟩, probability is (1 + |⟨ψ|φ⟩|²)/2
    
    Args:
        n_qubits: Number of qubits per register
        
    Returns:
        Quantum circuit template for swap test
    """
    # Create registers
    ancilla = QuantumRegister(1, 'ancilla')
    reg1 = QuantumRegister(n_qubits, 'reg1')
    reg2 = QuantumRegister(n_qubits, 'reg2')
    creg = ClassicalRegister(1, 'c')
    
    qc = QuantumCircuit(ancilla, reg1, reg2, creg)
    
    # Step 1: Hadamard on ancilla
    qc.h(ancilla[0])
    
    # Step 2: Controlled-SWAP between reg1 and reg2
    for i in range(n_qubits):
        qc.cswap(ancilla[0], reg1[i], reg2[i])
    
    # Step 3: Hadamard on ancilla
    qc.h(ancilla[0])
    
    # Step 4: Measure ancilla
    qc.measure(ancilla[0], creg[0])
    
    return qc


def swap_test_fidelity(
    vector1: np.ndarray,
    vector2: np.ndarray,
    shots: int = 1024,
    backend_name: str = 'qasm_simulator'
) -> Tuple[float, dict]:
    """
    Compute fidelity between two vectors using quantum swap test.
    
    Fidelity: F = |⟨ψ|φ⟩|²
    
    From swap test:
        P(ancilla=0) = (1 + F) / 2
    Therefore:
        F = 2 * P(0) - 1
    
    Args:
        vector1: First vector (d,)
        vector2: Second vector (d,)
        shots: Number of measurement shots
        backend_name: Qiskit backend to use
        
    Returns:
        (fidelity, result_dict) where result_dict contains measurement counts
    """
    # Normalize vectors
    v1_norm = normalize_vector(vector1)
    v2_norm = normalize_vector(vector2)
    
    # Get dimension and required qubits
    d = len(vector1)
    n_qubits = int(np.ceil(np.log2(d)))
    
    # Create full circuit
    # First, encode the two states
    state1_circuit = amplitude_encode(v1_norm, normalize=False)
    state2_circuit = amplitude_encode(v2_norm, normalize=False)
    
    # Create swap test circuit
    ancilla = QuantumRegister(1, 'ancilla')
    reg1 = QuantumRegister(n_qubits, 'reg1')
    reg2 = QuantumRegister(n_qubits, 'reg2')
    creg = ClassicalRegister(1, 'c')
    
    qc = QuantumCircuit(ancilla, reg1, reg2, creg)
    
    # Initialize states
    qc.compose(state1_circuit, qubits=reg1, inplace=True)
    qc.compose(state2_circuit, qubits=reg2, inplace=True)
    
    # Apply swap test
    qc.h(ancilla[0])
    for i in range(n_qubits):
        qc.cswap(ancilla[0], reg1[i], reg2[i])
    qc.h(ancilla[0])
    
    # Measure ancilla
    qc.measure(ancilla[0], creg[0])
    
    # Run on simulator
    backend = Aer.get_backend(backend_name)
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate fidelity from measurement
    prob_0 = counts.get('0', 0) / shots
    fidelity = 2 * prob_0 - 1
    
    # Fidelity should be in [0, 1], but due to sampling noise it might be slightly off
    fidelity = np.clip(fidelity, 0.0, 1.0)
    
    result_dict = {
        'counts': counts,
        'prob_0': prob_0,
        'shots': shots,
        'circuit_depth': qc.depth(),
        'num_qubits': qc.num_qubits
    }
    
    return fidelity, result_dict


def quantum_frame_similarity(
    frame1: np.ndarray,
    frame2: np.ndarray,
    shots: int = 1024
) -> float:
    """
    Compute quantum similarity between two frames.
    
    Returns fidelity F ∈ [0, 1] where:
    - F = 1: identical frames
    - F = 0: orthogonal frames
    
    Args:
        frame1: First frame (d,)
        frame2: Second frame (d,)
        shots: Number of measurement shots
        
    Returns:
        Fidelity value
    """
    fidelity, _ = swap_test_fidelity(frame1, frame2, shots=shots)
    return fidelity


def quantum_frame_distance(
    frame1: np.ndarray,
    frame2: np.ndarray,
    shots: int = 1024
) -> float:
    """
    Compute quantum distance between two frames.
    
    Distance: δ_Q = 1 - F
    
    where F is the fidelity from swap test.
    
    Args:
        frame1: First frame (d,)
        frame2: Second frame (d,)
        shots: Number of measurement shots
        
    Returns:
        Quantum distance δ_Q ∈ [0, 1]
    """
    fidelity = quantum_frame_similarity(frame1, frame2, shots=shots)
    return 1.0 - fidelity


class QuantumSimilarityComputer:
    """
    Reusable quantum similarity computer for DTW.
    """
    
    def __init__(self, shots: int = 1024, backend_name: str = 'qasm_simulator'):
        """
        Initialize quantum similarity computer.
        
        Args:
            shots: Number of measurement shots per comparison
            backend_name: Qiskit backend
        """
        self.shots = shots
        self.backend = Aer.get_backend(backend_name)
        self.num_comparisons = 0
        
        logger.info(f"QuantumSimilarityComputer initialized with {shots} shots")
    
    def compute_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute quantum similarity (fidelity) between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Fidelity F ∈ [0, 1]
        """
        fidelity, _ = swap_test_fidelity(frame1, frame2, shots=self.shots)
        self.num_comparisons += 1
        return fidelity
    
    def compute_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute quantum distance between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Distance δ_Q ∈ [0, 1]
        """
        return 1.0 - self.compute_similarity(frame1, frame2)
    
    def get_stats(self) -> dict:
        """Get statistics about quantum computations."""
        return {
            'num_comparisons': self.num_comparisons,
            'shots_per_comparison': self.shots,
            'total_shots': self.num_comparisons * self.shots
        }


if __name__ == "__main__":
    # Test swap test
    np.random.seed(42)
    
    # Test 1: Identical vectors (should give F ≈ 1)
    v1 = np.array([0.6, 0.8])
    v2 = np.array([0.6, 0.8])
    
    fidelity, result = swap_test_fidelity(v1, v2, shots=2048)
    print(f"Test 1 - Identical vectors:")
    print(f"  Fidelity: {fidelity:.4f} (expected ≈ 1.0)")
    print(f"  Counts: {result['counts']}")
    
    # Test 2: Orthogonal vectors (should give F ≈ 0)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    
    fidelity, result = swap_test_fidelity(v1, v2, shots=2048)
    print(f"\nTest 2 - Orthogonal vectors:")
    print(f"  Fidelity: {fidelity:.4f} (expected ≈ 0.0)")
    print(f"  Counts: {result['counts']}")
    
    # Test 3: Similar but not identical
    v1 = np.array([0.6, 0.8])
    v2 = np.array([0.8, 0.6])
    
    fidelity, result = swap_test_fidelity(v1, v2, shots=2048)
    print(f"\nTest 3 - Similar vectors:")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  Distance: {1-fidelity:.4f}")
    print(f"  Counts: {result['counts']}")
    
    # Test with quantum computer
    print(f"\n" + "="*50)
    print("Testing QuantumSimilarityComputer:")
    
    qsc = QuantumSimilarityComputer(shots=1024)
    
    v1 = np.random.randn(8)
    v2 = np.random.randn(8)
    
    sim = qsc.compute_similarity(v1, v2)
    dist = qsc.compute_distance(v1, v2)
    
    print(f"  Similarity: {sim:.4f}")
    print(f"  Distance: {dist:.4f}")
    print(f"  Stats: {qsc.get_stats()}")
