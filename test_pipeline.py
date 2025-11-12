"""
Quick test of the full pipeline with minimal samples
"""
import sys
import logging

# Setup minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

from main import QuantumDTWPipeline

def test_pipeline():
    print("=" * 60)
    print("Testing Full Quantum DTW Pipeline (Quick Version)")
    print("=" * 60)
    
    try:
        # Initialize with smaller parameters for quick testing
        print("\n1. Initializing pipeline...")
        pipeline = QuantumDTWPipeline(
            data_dir="msr_action_data",
            d_q=8,          # 8D after PCA
            window=5,       # Smaller window for speed
            shots=256,      # Fewer shots for speed
            qaoa_p=1        # 1 layer QAOA for speed
        )
        
        print("   ✓ Pipeline initialized")
        
        # Load data
        print("\n2. Loading and preprocessing data...")
        pipeline.load_data()
        print(f"   ✓ Loaded {len(pipeline.train_sequences)} train sequences")
        print(f"   ✓ Loaded {len(pipeline.test_sequences)} test sequences")
        
        # Reduce to 8D
        print("\n3. Applying PCA projection (60D → 8D)...")
        pipeline.apply_pca()
        print(f"   ✓ Train sequences projected to {pipeline.train_sequences_pca[0].shape}")
        print(f"   ✓ Test sequences projected to {pipeline.test_sequences_pca[0].shape}")
        
        # Classical baseline
        print("\n4. Running classical DTW baseline...")
        classical_results = pipeline.run_classical_baseline()
        print(f"   ✓ Classical accuracy: {classical_results['accuracy']*100:.2f}%")
        print(f"   ✓ Total time: {classical_results['total_time']:.2f}s")
        
        # Quantum similarity (just 3 samples for quick test)
        print("\n5. Testing quantum similarity computation (3 samples)...")
        quantum_sim_results = pipeline.run_quantum_similarity_experiment(n_samples=3)
        print(f"   ✓ Quantum similarities computed for {len(quantum_sim_results['pairs'])} pairs")
        print(f"   ✓ Mean quantum cost: {quantum_sim_results['summary']['mean_quantum_cost']:.4f}")
        
        # QAOA refinement (just 2 samples for quick test)
        print("\n6. Testing QAOA path refinement (2 samples)...")
        qaoa_results = pipeline.run_qaoa_refinement_experiment(n_samples=2)
        print(f"   ✓ QAOA refinements completed for {len(qaoa_results['pairs'])} pairs")
        if 'summary' in qaoa_results and 'mean_path_improvement' in qaoa_results['summary']:
            print(f"   ✓ Mean path improvement: {qaoa_results['summary']['mean_path_improvement']:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 Full Pipeline Test PASSED!")
        print("=" * 60)
        print("\n✓ All components integrated successfully:")
        print("  - Data loading and preprocessing")
        print("  - PCA dimensionality reduction")
        print("  - Classical DTW baseline")
        print("  - Quantum similarity computation")
        print("  - QAOA path optimization")
        print("\nResults saved to:", pipeline.results_dir)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
