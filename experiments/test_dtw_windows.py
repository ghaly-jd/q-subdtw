"""
Test different DTW window sizes to find optimal configuration.
"""
import numpy as np
import time
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import MSRAction3D
from src.classical.dtw_distance import DTWDistance
from src.preprocessing.pca_reducer import SkeletonPCA

def test_window_size(window_size, num_samples=50):
    """Test DTW classifier with specific window size."""
    print(f"\n{'='*60}")
    print(f"Testing window_size={window_size}")
    print(f"{'='*60}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "msr_action_data"
    dataset = MSRAction3D(str(data_dir))
    train_seq, test_seq = dataset.get_cross_subject_split()
    
    print(f"Training samples: {len(train_seq)}")
    print(f"Test samples: {len(test_seq)}")
    
    # PCA reduction
    pca = SkeletonPCA(n_components=8)
    train_features = [seq.get_features() for seq in train_seq]
    test_features = [seq.get_features() for seq in test_seq[:num_samples]]
    
    pca.fit(train_features)
    train_reduced = [pca.apply_pca(f) for f in train_features]
    test_reduced = [pca.apply_pca(f) for f in test_features]
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # DTW classifier
    dtw = DTWDistance(window=window_size)
    
    # Test
    correct = 0
    total_time = 0
    
    for i, test_feat in enumerate(test_reduced):
        test_label = test_seq[i].action_id
        
        start_time = time.time()
        
        # Find nearest neighbor
        min_dist = float('inf')
        pred_label = None
        
        for j, train_feat in enumerate(train_reduced):
            dist = dtw.distance(test_feat, train_feat)
            if dist < min_dist:
                min_dist = dist
                pred_label = train_seq[j].action_id
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if pred_label == test_label:
            correct += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples} samples, "
                  f"Accuracy so far: {correct/(i+1):.2%}")
    
    accuracy = correct / num_samples
    avg_time = total_time / num_samples
    
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{num_samples})")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time per Sample: {avg_time:.4f}s")
    
    return {
        'window_size': window_size,
        'accuracy': accuracy,
        'correct': correct,
        'total': num_samples,
        'total_time': total_time,
        'avg_time': avg_time
    }

def main():
    """Test multiple window sizes."""
    results = []
    
    # Test different window sizes
    windows = [10, 15, 20, None]  # None = no window constraint
    
    for window in windows:
        try:
            result = test_window_size(window, num_samples=50)
            results.append(result)
        except Exception as e:
            print(f"Error with window={window}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "window_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "dtw_window_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'best_window': max(results, key=lambda x: x['accuracy'])['window_size'],
                'best_accuracy': max(results, key=lambda x: x['accuracy'])['accuracy'],
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"Window={result['window_size']}: "
              f"Accuracy={result['accuracy']:.2%}, "
              f"Time={result['avg_time']:.4f}s/sample")
    
    print(f"\nBest configuration: window={max(results, key=lambda x: x['accuracy'])['window_size']}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
