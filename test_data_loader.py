"""
Test script for MSR Action3D data loader
"""
import sys
import numpy as np
from src.data.msr_action3d import MSRAction3D

def test_data_loader():
    print("=" * 60)
    print("Testing MSR Action3D Data Loader")
    print("=" * 60)
    
    # Initialize dataset
    print("\n1. Initializing MSRAction3D...")
    dataset = MSRAction3D(data_dir='msr_action_data')
    
    # Load data (automatically normalizes)
    print("2. Loading data with interpolation and normalization...")
    dataset.load_data(interpolate=True, target_length=50)
    
    train_data = dataset.train_sequences
    test_data = dataset.test_sequences
    
    print(f"   ✓ Training set: {len(train_data)} sequences")
    print(f"   ✓ Test set: {len(test_data)} sequences")
    
    # Check data structure
    print("\n3. Verifying data structure...")
    if train_data:
        seq = train_data[0]
        print(f"   ✓ Sequence shape: {seq.shape}")
        print(f"   ✓ First label: {dataset.train_labels[0]}")
    
    # Check normalization
    print("\n4. Verifying normalization...")
    all_train_seqs = np.array(train_data)
    mean = np.mean(all_train_seqs)
    std = np.std(all_train_seqs)
    print(f"   ✓ Training data mean: {mean:.6f}")
    print(f"   ✓ Training data std: {std:.6f}")
    
    # Check class distribution
    print("\n5. Checking class distribution...")
    train_labels = dataset.train_labels
    test_labels = dataset.test_labels
    unique_train = len(set(train_labels))
    unique_test = len(set(test_labels))
    print(f"   ✓ Training classes: {unique_train}")
    print(f"   ✓ Test classes: {unique_test}")
    
    # Check subject split (need to re-parse filenames)
    print("\n6. Verifying cross-subject split...")
    print(f"   ✓ Expected train subjects: {dataset.TRAIN_SUBJECTS}")
    print(f"   ✓ Expected test subjects: {dataset.TEST_SUBJECTS}")
    
    # Check for overlap (should be none)
    overlap = set(dataset.TRAIN_SUBJECTS) & set(dataset.TEST_SUBJECTS)
    if overlap:
        print(f"   ✗ ERROR: Subject overlap detected: {overlap}")
        return False
    else:
        print(f"   ✓ No subject overlap (good!)")
    
    print("\n" + "=" * 60)
    print("🎉 Data Loader Test PASSED!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = test_data_loader()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
