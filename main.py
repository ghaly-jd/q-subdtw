"""
Main Pipeline for Quantum DTW on MSR Action3D
End-to-end orchestration of the hybrid quantum-classical DTW pipeline.
"""

import sys
import logging
import json
import time
from pathlib import Path

import numpy as np

# Import our modules
from src.data.msr_action3d import load_msr_action3d
from src.dtw.core import DTWClassifier, dtw_distance
from src.subspace.pca import project_dataset
from src.quantum.swap_fidelity import QuantumSimilarityComputer
from src.dtw.window_extract import extract_subsequence_window, compute_local_cost_matrix
from src.quantum.dtw_qubo import build_dtw_qubo, decode_qubo_solution, validate_path, evaluate_path_cost
from src.quantum.qaoa_solver import QAOASolver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('q_dtw_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QuantumDTWPipeline:
    """
    Complete hybrid quantum-classical DTW pipeline.
    """
    
    def __init__(
        self,
        data_dir: str = "msr_action_data",
        d_q: int = 8,
        window: int = 10,
        shots: int = 512,
        qaoa_p: int = 2,
        results_dir: str = "results"
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Path to MSR Action3D data
            d_q: Target quantum dimension
            window: DTW Sakoe-Chiba band width
            shots: Quantum circuit shots
            qaoa_p: QAOA depth
            results_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.d_q = d_q
        self.window = window
        self.shots = shots
        self.qaoa_p = qaoa_p
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.train_sequences = None
        self.train_labels = None
        self.test_sequences = None
        self.test_labels = None
        
        self.train_sequences_pca = None
        self.test_sequences_pca = None
        self.pca_model = None
        
        # Quantum components
        self.quantum_similarity = None
        self.qaoa_solver = None
        
        logger.info(f"QuantumDTWPipeline initialized: d_q={d_q}, window={window}")
    
    def load_data(self, interpolate: bool = True, target_length: int = 50):
        """Load and preprocess MSR Action3D dataset."""
        logger.info("="*60)
        logger.info("STEP 1: Loading MSR Action3D Dataset")
        logger.info("="*60)
        
        self.train_sequences, self.train_labels, self.test_sequences, self.test_labels = load_msr_action3d(
            data_dir=self.data_dir,
            interpolate=interpolate,
            target_length=target_length,
            save_split=True
        )
        
        logger.info(f"Loaded {len(self.train_sequences)} train, {len(self.test_sequences)} test sequences")
        logger.info(f"Original dimension: {self.train_sequences[0].shape[1]}")
    
    def apply_pca(self):
        """Apply PCA dimensionality reduction."""
        logger.info("="*60)
        logger.info(f"STEP 2: PCA Projection (60D → {self.d_q}D)")
        logger.info("="*60)
        
        self.train_sequences_pca, self.test_sequences_pca, self.pca_model = project_dataset(
            self.train_sequences,
            self.test_sequences,
            n_components=self.d_q
        )
        
        logger.info(f"Projected to {self.d_q}D")
    
    def run_classical_baseline(self) -> dict:
        """Run classical DTW baseline."""
        logger.info("="*60)
        logger.info("STEP 3: Classical DTW Baseline")
        logger.info("="*60)
        
        # Use PCA-projected data for fair comparison
        classifier = DTWClassifier(window=self.window)
        classifier.fit(self.train_sequences_pca, self.train_labels)
        
        # Evaluate on small test subset for speed
        n_test = min(50, len(self.test_sequences_pca))
        logger.info(f"Evaluating on {n_test} test samples...")
        
        results = classifier.evaluate(
            self.test_sequences_pca[:n_test],
            self.test_labels[:n_test]
        )
        
        # Save results
        self._save_results("classical_dtw_baseline.json", results)
        
        return results
    
    def initialize_quantum_components(self):
        """Initialize quantum similarity computer and QAOA solver."""
        logger.info("="*60)
        logger.info("STEP 4: Initialize Quantum Components")
        logger.info("="*60)
        
        self.quantum_similarity = QuantumSimilarityComputer(shots=self.shots)
        self.qaoa_solver = QAOASolver(p=self.qaoa_p, shots=self.shots, maxiter=30)
        
        logger.info(f"Quantum components initialized: shots={self.shots}, QAOA_p={self.qaoa_p}")
    
    def run_quantum_similarity_experiment(self, n_samples: int = 10) -> dict:
        """
        Experiment 2: Quantum similarity on DTW paths.
        
        For random pairs:
        1. Compute classical DTW path
        2. Compute quantum similarity on path cells
        3. Compare costs
        """
        logger.info("="*60)
        logger.info(f"STEP 5: Quantum Similarity Experiment (n={n_samples})")
        logger.info("="*60)
        
        if self.quantum_similarity is None:
            self.initialize_quantum_components()
        
        results = {
            'pairs': [],
            'classical_costs': [],
            'quantum_costs': [],
            'cost_differences': []
        }
        
        np.random.seed(42)
        for i in range(n_samples):
            # Pick random pair
            idx1 = np.random.randint(len(self.train_sequences_pca))
            idx2 = np.random.randint(len(self.train_sequences_pca))
            
            seq1 = self.train_sequences_pca[idx1]
            seq2 = self.train_sequences_pca[idx2]
            
            logger.info(f"\nPair {i+1}/{n_samples}: sequences {idx1} and {idx2}")
            
            # Classical DTW
            classical_dist, path = dtw_distance(
                seq1, seq2, window=self.window, return_path=True
            )
            
            logger.info(f"  Classical DTW distance: {classical_dist:.4f}")
            logger.info(f"  Path length: {len(path)}")
            
            # Compute quantum cost along path (sample a few points)
            quantum_cost = 0.0
            n_path_samples = min(10, len(path))
            sampled_indices = np.linspace(0, len(path)-1, n_path_samples, dtype=int)
            
            for idx in sampled_indices:
                i, j = path[idx]
                frame1 = seq1[i-1]  # Convert to 0-indexed
                frame2 = seq2[j-1]
                
                # Quantum distance
                q_dist = self.quantum_similarity.compute_distance(frame1, frame2)
                quantum_cost += q_dist
            
            # Scale to full path
            quantum_cost = quantum_cost * len(path) / n_path_samples
            
            logger.info(f"  Quantum cost: {quantum_cost:.4f}")
            logger.info(f"  Difference: {abs(classical_dist - quantum_cost):.4f}")
            
            results['pairs'].append((idx1, idx2))
            results['classical_costs'].append(float(classical_dist))
            results['quantum_costs'].append(float(quantum_cost))
            results['cost_differences'].append(float(abs(classical_dist - quantum_cost)))
        
        # Summary statistics
        results['summary'] = {
            'mean_classical_cost': float(np.mean(results['classical_costs'])),
            'mean_quantum_cost': float(np.mean(results['quantum_costs'])),
            'mean_difference': float(np.mean(results['cost_differences'])),
            'correlation': float(np.corrcoef(results['classical_costs'], results['quantum_costs'])[0, 1])
        }
        
        logger.info(f"\nSummary:")
        logger.info(f"  Mean classical cost: {results['summary']['mean_classical_cost']:.4f}")
        logger.info(f"  Mean quantum cost: {results['summary']['mean_quantum_cost']:.4f}")
        logger.info(f"  Mean difference: {results['summary']['mean_difference']:.4f}")
        logger.info(f"  Correlation: {results['summary']['correlation']:.4f}")
        
        self._save_results("quantum_similarity_experiment.json", results)
        
        return results
    
    def run_qaoa_refinement_experiment(self, n_samples: int = 5) -> dict:
        """
        Experiment 3: QAOA path refinement.
        
        For random pairs:
        1. Compute classical DTW
        2. Extract window around path
        3. Run QAOA to refine
        4. Compare paths and costs
        """
        logger.info("="*60)
        logger.info(f"STEP 6: QAOA Path Refinement (n={n_samples})")
        logger.info("="*60)
        
        if self.qaoa_solver is None:
            self.initialize_quantum_components()
        
        results = {
            'pairs': [],
            'classical_costs': [],
            'qaoa_costs': [],
            'improvements': [],
            'valid_paths': []
        }
        
        np.random.seed(123)
        for i in range(n_samples):
            # Pick random pair
            idx1 = np.random.randint(len(self.train_sequences_pca))
            idx2 = np.random.randint(len(self.train_sequences_pca))
            
            seq1 = self.train_sequences_pca[idx1]
            seq2 = self.train_sequences_pca[idx2]
            
            logger.info(f"\nPair {i+1}/{n_samples}: sequences {idx1} and {idx2}")
            
            # Classical DTW
            classical_dist, path = dtw_distance(
                seq1, seq2, window=self.window, return_path=True
            )
            
            logger.info(f"  Classical DTW distance: {classical_dist:.4f}")
            
            # Extract window for QAOA
            try:
                sub_q, sub_c, mini_path, window_cells, offsets = extract_subsequence_window(
                    seq1, seq2, path,
                    window_length=12,  # Small for tractability
                    band_width=2
                )
                
                # Compute costs (using Euclidean for simplicity)
                costs = compute_local_cost_matrix(sub_q, sub_c, window_cells)
                
                logger.info(f"  Window: {len(window_cells)} cells")
                
                # Build QUBO
                qubo, var_mapping = build_dtw_qubo(
                    window_cells, costs,
                    T_q=len(sub_q), T_c=len(sub_c),
                    penalty_weight=5.0
                )
                
                # Solve with QAOA
                logger.info("  Running QAOA...")
                best_bitstring, best_energy, qaoa_results = self.qaoa_solver.solve(qubo)
                
                # Decode solution
                qaoa_path = decode_qubo_solution(best_bitstring, var_mapping)
                is_valid, msg = validate_path(qaoa_path, len(sub_q), len(sub_c))
                
                logger.info(f"  QAOA path valid: {is_valid}, {msg}")
                logger.info(f"  QAOA energy: {best_energy:.4f}")
                
                # Evaluate QAOA path cost
                qaoa_cost = evaluate_path_cost(qaoa_path, costs)
                
                # Classical path cost in window
                classical_window_cost = evaluate_path_cost(mini_path, costs)
                
                improvement = classical_window_cost - qaoa_cost
                
                logger.info(f"  Classical window cost: {classical_window_cost:.4f}")
                logger.info(f"  QAOA cost: {qaoa_cost:.4f}")
                logger.info(f"  Improvement: {improvement:.4f}")
                
                results['pairs'].append((idx1, idx2))
                results['classical_costs'].append(float(classical_window_cost))
                results['qaoa_costs'].append(float(qaoa_cost))
                results['improvements'].append(float(improvement))
                results['valid_paths'].append(is_valid)
                
            except Exception as e:
                logger.error(f"  Error in QAOA refinement: {e}")
                continue
        
        # Summary
        if results['improvements']:
            results['summary'] = {
                'mean_classical_cost': float(np.mean(results['classical_costs'])),
                'mean_qaoa_cost': float(np.mean(results['qaoa_costs'])),
                'mean_improvement': float(np.mean(results['improvements'])),
                'num_improvements': int(np.sum(np.array(results['improvements']) > 0)),
                'num_valid_paths': int(np.sum(results['valid_paths']))
            }
            
            logger.info(f"\nSummary:")
            logger.info(f"  Mean improvement: {results['summary']['mean_improvement']:.4f}")
            logger.info(f"  Paths with improvement: {results['summary']['num_improvements']}/{len(results['improvements'])}")
            logger.info(f"  Valid paths: {results['summary']['num_valid_paths']}/{len(results['valid_paths'])}")
        
        self._save_results("qaoa_refinement_experiment.json", results)
        
        return results
    
    def _save_results(self, filename: str, results: dict):
        """Save results to JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        logger.info("\n" + "="*60)
        logger.info("QUANTUM DTW PIPELINE - FULL RUN")
        logger.info("="*60 + "\n")
        
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: PCA projection
        self.apply_pca()
        
        # Step 3: Classical baseline
        classical_results = self.run_classical_baseline()
        
        # Step 4: Quantum experiments
        quantum_sim_results = self.run_quantum_similarity_experiment(n_samples=10)
        qaoa_results = self.run_qaoa_refinement_experiment(n_samples=5)
        
        total_time = time.time() - start_time
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Classical accuracy: {classical_results['accuracy']*100:.2f}%")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    pipeline = QuantumDTWPipeline(
        data_dir="msr_action_data",
        d_q=8,
        window=10,
        shots=512,
        qaoa_p=2
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
