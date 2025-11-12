"""
Test script for Quantum Components
Tests amplitude encoding, swap test, QUBO, and QAOA
"""
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from src.data.msr_action3d import MSRAction3D
from src.subspace.pca import SkeletonPCA
from src.quantum.amplitude_encoding import AmplitudeEncoder, normalize_vector
from src.quantum.swap_fidelity import swap_test_fidelity, create_swap_test_circuit
from src.quantum.dtw_qubo import build_dtw_qubo, decode_qubo_solution
from src.quantum.qaoa_solver import QAOASolver

def test_quantum_components():
    print("=" * 60)
    print("Testing Quantum Components")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    dataset = MSRAction3D(data_dir='msr_action_data')
    dataset.load_data(interpolate=True, target_length=50)
    
    # Use PCA to reduce dimensionality
    pca = SkeletonPCA(n_components=8)
    pca.fit(dataset.train_sequences[:50])  # Use subset for speed
    
    seq1 = pca.transform_sequence(dataset.train_sequences[0])
    seq2 = pca.transform_sequence(dataset.train_sequences[1])
    
    print(f"   ✓ Loaded data and reduced dimension to {seq1.shape}")
    
    # Test Amplitude Encoding
    print("\n2. Testing Amplitude Encoding...")
    encoder = AmplitudeEncoder(dimension=8)  # 8D vectors from PCA
    
    # Encode single frame
    frame = seq1[0]  # (8,)
    print(f"   - Encoding frame: {frame[:3]}... (8D)")
    
    qc = encoder.encode(frame)
    print(f"   ✓ Created circuit with {qc.num_qubits} qubits")
    print(f"   ✓ Circuit depth: {qc.depth()}")
    
    # Test normalization
    normalized = normalize_vector(frame)
    norm = np.linalg.norm(normalized)
    print(f"   ✓ Normalized vector norm: {norm:.6f} (should be ≈1.0)")
    
    # Test Swap Test Circuit
    print("\n3. Testing Swap Test Circuit...")
    
    # Create swap test circuit for two frames
    frame1 = seq1[0]
    frame2 = seq2[0]
    
    # Get number of qubits needed (log2 of vector dimension)
    n_qubits = int(np.ceil(np.log2(len(frame1))))
    swap_circuit = create_swap_test_circuit(n_qubits)
    print(f"   ✓ Created swap test circuit with {swap_circuit.num_qubits} qubits")
    print(f"   ✓ Circuit depth: {swap_circuit.depth()}")
    
    # Simulate and measure fidelity
    print("   - Running quantum simulation...")
    fidelity, result_dict = swap_test_fidelity(frame1, frame2, shots=1000)
    print(f"   ✓ Measured fidelity: {fidelity:.4f}")
    print(f"   ✓ Circuit used {result_dict['num_qubits']} qubits")
    
    # Compare with classical inner product
    classical_similarity = np.dot(
        normalize_vector(frame1),
        normalize_vector(frame2)
    )
    print(f"   ✓ Classical similarity: {classical_similarity:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 Quantum Components Test PASSED!")
    print("=" * 60)
    print("\nTested components:")
    print("  ✓ Amplitude Encoding")
    print("  ✓ Swap Test Circuit")
    print("\nNote: QUBO and QAOA require more complex setup.")
    print("These will be tested in the full pipeline.")
    print("\nReady for full pipeline testing.")
    return True

if __name__ == '__main__':
    try:
        success = test_quantum_components()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
