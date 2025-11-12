"""
Quick Improvements Test - Tests key improvements systematically
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import QuantumDTWPipeline
import json

def main():
    print("="*70)
    print("QUICK IMPROVEMENTS TEST")
    print("="*70)
    
    # Test 1: Baseline with original window
    print("\n1. Testing baseline (window=10)...")
    pipeline = QuantumDTWPipeline("msr_action_data")
    results_baseline = pipeline.run_experiments(num_test_samples=30, num_quantum_pairs=20)
    
    # Test 2: Larger window
    print("\n2. Testing larger window (window=15)...")
    pipeline_w15 = QuantumDTWPipeline("msr_action_data", dtw_window=15)
    results_w15 = pipeline_w15.run_experiments(num_test_samples=30, num_quantum_pairs=20)
    
    # Test 3: Even larger window
    print("\n3. Testing even larger window (window=20)...")
    pipeline_w20 = QuantumDTWPipeline("msr_action_data", dtw_window=20)
    results_w20 = pipeline_w20.run_experiments(num_test_samples=30, num_quantum_pairs=20)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "improvements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'baseline_window10': {
            'accuracy': results_baseline.get('classical_accuracy', 0),
            'correlation': results_baseline.get('quantum_correlation', 0)
        },
        'improved_window15': {
            'accuracy': results_w15.get('classical_accuracy', 0),
            'correlation': results_w15.get('quantum_correlation', 0)
        },
        'improved_window20': {
            'accuracy': results_w20.get('classical_accuracy', 0),
            'correlation': results_w20.get('quantum_correlation', 0)
        }
    }
    
    output_file = output_dir / "quick_improvements.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Window=10: Accuracy={all_results['baseline_window10']['accuracy']:.2%}, "
          f"Correlation={all_results['baseline_window10']['correlation']:.4f}")
    print(f"Window=15: Accuracy={all_results['improved_window15']['accuracy']:.2%}, "
          f"Correlation={all_results['improved_window15']['correlation']:.4f}")
    print(f"Window=20: Accuracy={all_results['improved_window20']['accuracy']:.2%}, "
          f"Correlation={all_results['improved_window20']['correlation']:.4f}")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
