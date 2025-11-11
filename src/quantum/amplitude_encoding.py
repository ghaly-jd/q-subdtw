"""
Amplitude Encoding for Quantum States
Prepares quantum states from classical vectors using amplitude encoding.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Initialize
import logging

logger = logging.getLogger(__name__)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        logger.warning("Near-zero vector encountered in normalization")
        # Return uniform distribution
        return np.ones_like(vector) / np.sqrt(len(vector))
    return vector / norm


def pad_to_power_of_2(vector: np.ndarray) -> np.ndarray:
    """
    Pad vector to the nearest power of 2 length.
    
    For amplitude encoding, we need 2^n amplitudes where n is the number of qubits.
    
    Args:
        vector: Input vector of length d
        
    Returns:
        Padded vector of length 2^n where n = ceil(log2(d))
    """
    d = len(vector)
    n_qubits = int(np.ceil(np.log2(d)))
    target_len = 2 ** n_qubits
    
    if d == target_len:
        return vector
    
    # Pad with zeros
    padded = np.zeros(target_len)
    padded[:d] = vector
    
    return padded


def amplitude_encode(vector: np.ndarray, normalize: bool = True) -> QuantumCircuit:
    """
    Create quantum circuit that encodes a classical vector as amplitudes.
    
    Given a d-dimensional vector x, we prepare a quantum state:
        |ψ⟩ = Σ_i x_i |i⟩
    
    where x_i are normalized to satisfy Σ_i |x_i|^2 = 1.
    
    Args:
        vector: Classical vector to encode (d,)
        normalize: Whether to normalize the vector
        
    Returns:
        Quantum circuit with the state prepared
    """
    # Normalize if requested
    if normalize:
        vector = normalize_vector(vector)
    
    # Pad to power of 2
    padded = pad_to_power_of_2(vector)
    n_qubits = int(np.log2(len(padded)))
    
    # Create circuit
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    
    # Use Qiskit's Initialize instruction
    qc.initialize(padded, qr)
    
    return qc


def get_num_qubits(dimension: int) -> int:
    """
    Calculate number of qubits needed for amplitude encoding.
    
    Args:
        dimension: Dimension of classical vector
        
    Returns:
        Number of qubits needed (ceil(log2(dimension)))
    """
    return int(np.ceil(np.log2(dimension)))


class AmplitudeEncoder:
    """
    Reusable amplitude encoder for sequences.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize encoder for given dimension.
        
        Args:
            dimension: Dimension of vectors to encode
        """
        self.dimension = dimension
        self.n_qubits = get_num_qubits(dimension)
        self.padded_dimension = 2 ** self.n_qubits
        
        logger.info(f"AmplitudeEncoder: {dimension}D → {self.n_qubits} qubits "
                   f"(padded to {self.padded_dimension})")
    
    def encode(self, vector: np.ndarray, normalize: bool = True) -> QuantumCircuit:
        """
        Encode a vector into a quantum circuit.
        
        Args:
            vector: Vector to encode (must be of length self.dimension)
            normalize: Whether to normalize
            
        Returns:
            Quantum circuit
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Expected vector of length {self.dimension}, got {len(vector)}")
        
        return amplitude_encode(vector, normalize=normalize)
    
    def encode_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple:
        """
        Encode two frames for comparison (used in swap test).
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            (circuit1, circuit2)
        """
        return self.encode(frame1), self.encode(frame2)


if __name__ == "__main__":
    # Test amplitude encoding
    
    # Small vector
    v = np.array([0.6, 0.8])
    qc = amplitude_encode(v)
    print(f"Vector: {v}")
    print(f"Qubits needed: {qc.num_qubits}")
    print(f"\nCircuit:")
    print(qc)
    
    # Larger vector (typical quantum dimension)
    print("\n" + "="*50)
    d_q = 8
    v = np.random.randn(d_q)
    
    encoder = AmplitudeEncoder(dimension=d_q)
    qc = encoder.encode(v)
    
    print(f"\nEncoding {d_q}D vector:")
    print(f"  Qubits: {encoder.n_qubits}")
    print(f"  Padded dimension: {encoder.padded_dimension}")
    print(f"  Circuit depth: {qc.depth()}")
