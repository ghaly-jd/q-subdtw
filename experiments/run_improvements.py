"""
Comprehensive improvement experiments.

Tests all improvements systematically:
1. Baseline (original)
2. DTW window variations
3. Multi-scale quantum (50 pairs)
4. Hybrid quantum-classical DTW
5. QAOA refinement (with fix)
"""
import numpy as np
import json
import time
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from main
from main import QuantumDTWPipeline


def load_and_prepare_data(num_test_samples=50):
    """Load MSR Action3D data and apply PCA."""
    print("Loading data...")
    data_dir = Path(__file__).parent.parent / "msr_action_data"
    dataset = MSRAction3D(str(data_dir))
    train_seq, test_seq = dataset.get_cross_subject_split()
    
    print(f"Training samples: {len(train_seq)}")
    print(f"Test samples: {len(test_seq)}")
    
    # PCA reduction
    print("Applying PCA...")
    pca = SkeletonPCA(n_components=8)
    train_features = [seq.get_features() for seq in train_seq]
    test_features = [seq.get_features() for seq in test_seq[:num_test_samples]]
    
    pca.fit(train_features)
    train_reduced = [pca.apply_pca(f) for f in train_features]
    test_reduced = [pca.apply_pca(f) for f in test_features]
    
    train_labels = [seq.action_id for seq in train_seq]
    test_labels = [seq.action_id for seq in test_seq[:num_test_samples]]
    
    print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    
    return train_reduced, test_reduced, train_labels, test_labels


def experiment_1_dtw_windows(train_data, test_data, train_labels, test_labels):
    """Test different DTW window sizes."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: DTW Window Size Comparison")
    print("="*70)
    
    windows = [10, 15, 20, None]
    results = []
    
    for window in windows:
        print(f"\nTesting window={window}...")
        dtw = DTWDistance(window=window)
        
        start_time = time.time()
        correct = 0
        
        for i, (test_feat, test_label) in enumerate(zip(test_data, test_labels)):
            # 1-NN classification
            min_dist = float('inf')
            pred_label = None
            
            for train_feat, train_label in zip(train_data, train_labels):
                dist = dtw.distance(test_feat, train_feat)
                if dist < min_dist:
                    min_dist = dist
                    pred_label = train_label
            
            if pred_label == test_label:
                correct += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_data)}, "
                      f"Accuracy: {correct/(i+1):.2%}")
        
        elapsed = time.time() - start_time
        accuracy = correct / len(test_data)
        
        result = {
            'window': window,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
            'time': elapsed,
            'avg_time_per_sample': elapsed / len(test_data)
        }
        results.append(result)
        
        print(f"  Final: Accuracy={accuracy:.2%}, Time={elapsed:.2f}s")
    
    return results


def experiment_2_multiscale_quantum(train_data, test_data, num_pairs=50):
    """Test multi-scale quantum similarity with more pairs."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multi-Scale Quantum Similarity (50 pairs)")
    print("="*70)
    
    # Select pairs from different and same classes
    np.random.seed(42)
    pairs = []
    
    # Sample pairs
    for _ in range(num_pairs):
        i = np.random.randint(0, len(test_data))
        j = np.random.randint(0, len(test_data))
        if i != j:
            pairs.append((i, j))
    
    pairs = pairs[:num_pairs]
    
    results = []
    dtw = DTWDistance(window=10)
    
    print(f"Computing {len(pairs)} pairs...")
    for idx, (i, j) in enumerate(pairs):
        # Classical DTW
        classical_cost = dtw.distance(test_data[i], test_data[j])
        
        # Multi-scale quantum
        vec1 = np.mean(test_data[i], axis=0)
        vec2 = np.mean(test_data[j], axis=0)
        
        quantum_cost, scale_details = multiscale_quantum_similarity(
            vec1, vec2,
            qubit_scales=[3, 4, 5],
            shots=1024
        )
        
        results.append({
            'pair': (i, j),
            'classical_cost': classical_cost,
            'quantum_cost': quantum_cost,
            'scale_details': scale_details
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(pairs)}")
    
    # Compute correlation
    classical_costs = [r['classical_cost'] for r in results]
    quantum_costs = [r['quantum_cost'] for r in results]
    
    correlation, p_value = pearsonr(classical_costs, quantum_costs)
    
    summary = {
        'num_pairs': len(results),
        'mean_classical_cost': np.mean(classical_costs),
        'mean_quantum_cost': np.mean(quantum_costs),
        'std_classical_cost': np.std(classical_costs),
        'std_quantum_cost': np.std(quantum_costs),
        'correlation': correlation,
        'p_value': p_value
    }
    
    print(f"\n  Correlation: {correlation:.4f} (p={p_value:.6f})")
    print(f"  Mean Classical: {summary['mean_classical_cost']:.2f}")
    print(f"  Mean Quantum: {summary['mean_quantum_cost']:.2f}")
    
    return {'pairs': results, 'summary': summary}


def experiment_3_hybrid_dtw(train_data, test_data, train_labels, test_labels):
    """Test hybrid quantum-classical DTW."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Hybrid Quantum-Classical DTW")
    print("="*70)
    
    alphas = [1.0, 0.8, 0.7, 0.5, 0.3, 0.0]  # 1.0=pure classical, 0.0=pure quantum
    results = []
    
    for alpha in alphas:
        print(f"\nTesting alpha={alpha}...")
        
        classifier = HybridClassifier(
            alpha=alpha,
            window=10,
            use_multiscale=True
        )
        classifier.fit(train_data, train_labels)
        
        start_time = time.time()
        correct = 0
        
        for i, (test_seq, test_label) in enumerate(zip(test_data, test_labels)):
            pred = classifier.predict(test_seq)
            if pred == test_label:
                correct += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_data)}, "
                      f"Accuracy: {correct/(i+1):.2%}")
        
        elapsed = time.time() - start_time
        accuracy = correct / len(test_data)
        
        result = {
            'alpha': alpha,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_data),
            'time': elapsed,
            'description': f"alpha={alpha} ({'pure classical' if alpha==1.0 else 'pure quantum' if alpha==0.0 else f'{int(alpha*100)}% classical'})"
        }
        results.append(result)
        
        print(f"  Final: Accuracy={accuracy:.2%}, Time={elapsed:.2f}s")
    
    return results


def experiment_4_qaoa_refinement(test_data, num_pairs=5):
    """Test QAOA refinement with fixed API."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: QAOA Refinement (Fixed API)")
    print("="*70)
    
    # Select pairs
    np.random.seed(42)
    pairs = [(np.random.randint(0, len(test_data)), 
              np.random.randint(0, len(test_data))) 
             for _ in range(num_pairs * 2)]
    pairs = [(i, j) for i, j in pairs if i != j][:num_pairs]
    
    qaoa = QAOASolver(p=2, maxiter=50)
    results = []
    
    print(f"Testing {len(pairs)} pairs with QAOA...")
    for idx, (i, j) in enumerate(pairs):
        print(f"\n  Pair {idx+1}/{len(pairs)}: ({i}, {j})")
        
        try:
            # Simplified QUBO: just a small test problem
            # In real use, this would be DTW alignment QUBO
            n_vars = 4
            qubo = {(i, i): -1.0 for i in range(n_vars)}
            qubo[(0, 1)] = 2.0
            qubo[(1, 2)] = 2.0
            
            start_time = time.time()
            best_bitstring, best_energy, qaoa_results = qaoa.solve(qubo, n_qubits=n_vars)
            elapsed = time.time() - start_time
            
            result = {
                'pair': (i, j),
                'success': True,
                'best_bitstring': best_bitstring,
                'best_energy': best_energy,
                'time': elapsed,
                'iterations': len(qaoa_results) if isinstance(qaoa_results, list) else 1
            }
            
            print(f"    Success! Energy={best_energy:.4f}, Time={elapsed:.2f}s")
            
        except Exception as e:
            print(f"    Error: {e}")
            result = {
                'pair': (i, j),
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n  Successful: {successful}/{len(results)}")
    
    return results


def main():
    """Run all experiments."""
    print("="*70)
    print("COMPREHENSIVE IMPROVEMENT EXPERIMENTS")
    print("="*70)
    
    # Load data
    train_data, test_data, train_labels, test_labels = load_and_prepare_data(
        num_test_samples=50
    )
    
    # Run all experiments
    all_results = {}
    
    # Experiment 1: DTW Windows
    all_results['dtw_windows'] = experiment_1_dtw_windows(
        train_data, test_data, train_labels, test_labels
    )
    
    # Experiment 2: Multi-scale Quantum
    all_results['multiscale_quantum'] = experiment_2_multiscale_quantum(
        train_data, test_data, num_pairs=50
    )
    
    # Experiment 3: Hybrid DTW
    all_results['hybrid_dtw'] = experiment_3_hybrid_dtw(
        train_data, test_data, train_labels, test_labels
    )
    
    # Experiment 4: QAOA
    all_results['qaoa_refinement'] = experiment_4_qaoa_refinement(
        test_data, num_pairs=5
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "improvements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "comprehensive_improvements.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n1. DTW Windows:")
    best_window = max(all_results['dtw_windows'], key=lambda x: x['accuracy'])
    for r in all_results['dtw_windows']:
        marker = " ← BEST" if r == best_window else ""
        print(f"   Window={r['window']}: {r['accuracy']:.2%}{marker}")
    
    print("\n2. Multi-scale Quantum:")
    summary = all_results['multiscale_quantum']['summary']
    print(f"   Correlation: {summary['correlation']:.4f} (p={summary['p_value']:.6f})")
    print(f"   Pairs tested: {summary['num_pairs']}")
    
    print("\n3. Hybrid DTW:")
    best_hybrid = max(all_results['hybrid_dtw'], key=lambda x: x['accuracy'])
    for r in all_results['hybrid_dtw']:
        marker = " ← BEST" if r == best_hybrid else ""
        print(f"   Alpha={r['alpha']}: {r['accuracy']:.2%}{marker}")
    
    print("\n4. QAOA Refinement:")
    qaoa_success = sum(1 for r in all_results['qaoa_refinement'] if r.get('success', False))
    print(f"   Successful: {qaoa_success}/{len(all_results['qaoa_refinement'])}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
