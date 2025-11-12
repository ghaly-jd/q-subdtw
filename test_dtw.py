"""
Test script for Classical DTW implementation
"""
import sys
import numpy as np
from src.data.msr_action3d import MSRAction3D
from src.dtw.core import dtw_distance, DTWClassifier

def test_dtw():
    print("=" * 60)
    print("Testing Classical DTW")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MSR Action3D dataset...")
    dataset = MSRAction3D(data_dir='msr_action_data')
    dataset.load_data(interpolate=True, target_length=50)
    
    print(f"   ✓ Training: {len(dataset.train_sequences)} sequences")
    print(f"   ✓ Test: {len(dataset.test_sequences)} sequences")
    
    # Test DTW distance function
    print("\n2. Testing DTW distance function...")
    seq1 = dataset.train_sequences[0]
    seq2 = dataset.train_sequences[1]
    
    # Test with different window sizes
    dist_full, _ = dtw_distance(seq1, seq2, window=None)
    dist_window, _ = dtw_distance(seq1, seq2, window=10)
    
    print(f"   ✓ DTW distance (full): {dist_full:.4f}")
    print(f"   ✓ DTW distance (window=10): {dist_window:.4f}")
    
    # Test DTW path return
    print("\n3. Testing DTW path extraction...")
    dist, path = dtw_distance(seq1, seq2, window=10, return_path=True)
    print(f"   ✓ Path length: {len(path)}")
    print(f"   ✓ Path start: {path[0]}")
    print(f"   ✓ Path end: {path[-1]}")
    
    # Test DTW Classifier with small subset
    print("\n4. Testing DTW Classifier (small subset)...")
    # Use only first 30 training samples for quick test
    small_train = dataset.train_sequences[:30]
    small_train_labels = dataset.train_labels[:30]
    small_test = dataset.test_sequences[:10]
    small_test_labels = dataset.test_labels[:10]
    
    classifier = DTWClassifier(window=10)
    print("   - Training classifier...")
    classifier.fit(small_train, small_train_labels)
    
    print("   - Making predictions...")
    predictions = classifier.predict(small_test)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(small_test_labels))
    print(f"   ✓ Accuracy on small test set: {accuracy*100:.2f}%")
    print(f"   ✓ Predictions: {predictions[:5]}")
    print(f"   ✓ True labels: {small_test_labels[:5]}")
    
    # Test single sequence prediction
    print("\n5. Testing single sequence prediction...")
    pred, dist = classifier.predict_one(small_test[0], return_distance=True)
    print(f"   ✓ Single prediction: {pred}")
    print(f"   ✓ Distance to nearest neighbor: {dist:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 Classical DTW Test PASSED!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = test_dtw()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
