"""
Test script for PCA dimensionality reduction
"""
import sys
import numpy as np
from src.data.msr_action3d import MSRAction3D
from src.subspace.pca import SkeletonPCA

def test_pca():
    print("=" * 60)
    print("Testing PCA Dimensionality Reduction")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MSR Action3D dataset...")
    dataset = MSRAction3D(data_dir='msr_action_data')
    dataset.load_data(interpolate=True, target_length=50)
    
    print(f"   ✓ Training: {len(dataset.train_sequences)} sequences")
    print(f"   ✓ Original dimension: {dataset.train_sequences[0].shape}")
    
    # Initialize PCA
    print("\n2. Initializing PCA projector (60D → 8D)...")
    pca = SkeletonPCA(n_components=8)
    
    # Fit on training data
    print("\n3. Fitting PCA on training data...")
    pca.fit(dataset.train_sequences)
    
    explained_var = np.sum(pca.pca.explained_variance_ratio_)
    print(f"   ✓ Explained variance ratio: {explained_var:.4f}")
    print(f"   ✓ PCA components shape: {pca.pca.components_.shape}")
    
    # Transform single sequence
    print("\n4. Testing single sequence transformation...")
    test_seq = dataset.train_sequences[0]
    projected = pca.transform_sequence(test_seq)
    
    print(f"   ✓ Original shape: {test_seq.shape}")
    print(f"   ✓ Projected shape: {projected.shape}")
    print(f"   ✓ First frame (original): {test_seq[0][:5]}... (60D)")
    print(f"   ✓ First frame (projected): {projected[0]} (8D)")
    
    # Transform batch
    print("\n5. Testing batch transformation...")
    test_batch = dataset.train_sequences[:10]
    projected_batch = pca.transform_sequences(test_batch)
    
    print(f"   ✓ Batch size: {len(test_batch)}")
    print(f"   ✓ Projected batch length: {len(projected_batch)}")
    print(f"   ✓ Each sequence shape: {projected_batch[0].shape}")
    
    # Verify dimensionality reduction
    print("\n6. Verifying dimensionality reduction...")
    original_size = test_seq.shape[0] * test_seq.shape[1]  # T x 60
    projected_size = projected.shape[0] * projected.shape[1]  # T x 8
    compression_ratio = original_size / projected_size
    
    print(f"   ✓ Original size: {original_size} values")
    print(f"   ✓ Projected size: {projected_size} values")
    print(f"   ✓ Compression ratio: {compression_ratio:.2f}x")
    
    # Test with different target dimensions
    print("\n7. Testing different target dimensions...")
    for n_comp in [4, 8, 16]:
        pca_test = SkeletonPCA(n_components=n_comp)
        pca_test.fit(dataset.train_sequences[:50])  # Use subset for speed
        proj = pca_test.transform_sequence(test_seq)
        var = np.sum(pca_test.pca.explained_variance_ratio_)
        print(f"   ✓ {n_comp}D: shape={proj.shape}, variance={var:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 PCA Test PASSED!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = test_pca()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
