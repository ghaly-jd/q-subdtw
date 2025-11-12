"""
Simplified Improvements Test
Tests: 1) QAOA fix, 2) Window sizes, 3) More quantum pairs
"""
import sys
import json
import time
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports that we know work
from main import QuantumDTWPipeline

def test_qaoa_fix():
    """Test if QAOA works now with assign_parameters fix."""
    print("\n" + "="*70)
    print("TEST 1: QAOA API Fix")
    print("="*70)
    
    try:
        from src.quantum.qaoa_solver import QAOASolver
        
        qaoa = QAOASolver(p=2, maxiter=30)
        
        # Simple test QUBO
        qubo = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        
        print("Running QAOA solver...")
        best_bits, best_energy, results = qaoa.solve(qubo, n_qubits=2)
        
        print(f"✓ QAOA SUCCESS!")
        print(f"  Best bitstring: {best_bits}")
        print(f"  Best energy: {best_energy:.4f}")
        
        return {'success': True, 'energy': best_energy}
        
    except Exception as e:
        print(f"✗ QAOA FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_window_sizes():
    """Test different DTW window sizes."""
    print("\n" + "="*70)
    print("TEST 2: DTW Window Sizes")
    print("="*70)
    
    windows = [10, 15, 20]
    results = []
    
    for window in windows:
        print(f"\nTesting window={window}...")
        try:
            pipeline = QuantumDTWPipeline(
                data_dir="msr_action_data",
                window=window
            )
            
            # Load data
            pipeline.load_data()
            pipeline.apply_pca()
            
            # Run classical DTW on small subset
            from src.classical.dtw_distance import DTWDistance
            dtw = DTWDistance(window=window)
            
            start_time = time.time()
            correct = 0
            num_test = 20  # Small sample for speed
            
            for i in range(num_test):
                test_seq = pipeline.test_sequences_pca[i]
                test_label = pipeline.test_labels[i]
                
                # 1-NN
                min_dist = float('inf')
                pred_label = None
                
                for j, train_seq in enumerate(pipeline.train_sequences_pca):
                    dist = dtw.distance(test_seq, train_seq)
                    if dist < min_dist:
                        min_dist = dist
                        pred_label = pipeline.train_labels[j]
                
                if pred_label == test_label:
                    correct += 1
            
            elapsed = time.time() - start_time
            accuracy = correct / num_test
            
            result = {
                'window': window,
                'accuracy': accuracy,
                'time': elapsed,
                'correct': correct,
                'total': num_test
            }
            results.append(result)
            
            print(f"  Accuracy: {accuracy:.2%}, Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'window': window, 'error': str(e)})
    
    return results


def test_quantum_pairs():
    """Test quantum similarity with 50 pairs."""
    print("\n" + "="*70)
    print("TEST 3: Quantum Similarity (50 pairs)")
    print("="*70)
    
    try:
        import numpy as np
        from scipy.stats import pearsonr
        from src.quantum.amplitude_encoding import amplitude_encode
        from src.quantum.swap_test import quantum_swap_test
        from src.classical.dtw_distance import DTWDistance
        
        pipeline = QuantumDTWPipeline(data_dir="msr_action_data")
        pipeline.load_data()
        pipeline.apply_pca()
        
        dtw = DTWDistance(window=10)
        
        # Sample 50 random pairs
        np.random.seed(42)
        num_pairs = 50
        pairs = []
        
        n_test = len(pipeline.test_sequences_pca)
        for _ in range(num_pairs * 2):
            i, j = np.random.randint(0, n_test), np.random.randint(0, n_test)
            if i != j:
                pairs.append((i, j))
        pairs = pairs[:num_pairs]
        
        print(f"Computing {len(pairs)} pairs...")
        results = []
        
        for idx, (i, j) in enumerate(pairs):
            # Classical
            classical_cost = dtw.distance(
                pipeline.test_sequences_pca[i],
                pipeline.test_sequences_pca[j]
            )
            
            # Quantum
            vec1 = np.mean(pipeline.test_sequences_pca[i], axis=0)
            vec2 = np.mean(pipeline.test_sequences_pca[j], axis=0)
            
            # Encode (3 qubits = 8 dimensions)
            state1 = amplitude_encode(vec1[:8], n_qubits=3)
            state2 = amplitude_encode(vec2[:8], n_qubits=3)
            similarity = quantum_swap_test(state1, state2, shots=1024)
            quantum_cost = 1.0 - similarity
            
            results.append({
                'pair': (i, j),
                'classical': classical_cost,
                'quantum': quantum_cost
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/{len(pairs)}")
        
        # Compute correlation
        classical_costs = [r['classical'] for r in results]
        quantum_costs = [r['quantum'] for r in results]
        
        correlation, p_value = pearsonr(classical_costs, quantum_costs)
        
        summary = {
            'num_pairs': len(results),
            'correlation': correlation,
            'p_value': p_value,
            'mean_classical': np.mean(classical_costs),
            'mean_quantum': np.mean(quantum_costs)
        }
        
        print(f"\n  Correlation: {correlation:.4f} (p={p_value:.6f})")
        print(f"  Mean classical: {summary['mean_classical']:.2f}")
        print(f"  Mean quantum: {summary['mean_quantum']:.2f}")
        
        return {'success': True, 'summary': summary, 'pairs': results}
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    """Run all improvement tests."""
    print("="*70)
    print("IMPROVEMENTS VALIDATION")
    print("="*70)
    
    all_results = {}
    
    # Test 1: QAOA Fix
    all_results['qaoa_fix'] = test_qaoa_fix()
    
    # Test 2: Window Sizes
    all_results['window_sizes'] = test_window_sizes()
    
    # Test 3: Quantum Pairs
    all_results['quantum_pairs'] = test_quantum_pairs()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "improvements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # QAOA
    if all_results['qaoa_fix']['success']:
        print("✓ QAOA API Fixed - assign_parameters working!")
    else:
        print("✗ QAOA still has issues")
    
    # Windows
    print("\nWindow Size Comparison:")
    if isinstance(all_results['window_sizes'], list):
        for r in all_results['window_sizes']:
            if 'accuracy' in r:
                print(f"  Window={r['window']}: {r['accuracy']:.2%}")
    
    # Quantum
    if all_results['quantum_pairs'].get('success'):
        summary = all_results['quantum_pairs']['summary']
        print(f"\nQuantum-Classical Correlation: {summary['correlation']:.4f}")
        print(f"(Based on {summary['num_pairs']} pairs)")
    
    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
