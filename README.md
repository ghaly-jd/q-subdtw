# Quantum DTW for Skeleton Action Recognition (Q-SubDTW)

A **hybrid quantum-classical pipeline** for action recognition using Dynamic Time Warping (DTW) on the MSR Action3D skeleton dataset. This implementation features genuine quantum computing components, not just quantum-inspired algorithms.

## 🎯 Project Overview

This project implements three quantum enhancements to classical DTW:

1. **Quantum Frame Similarity**: Uses swap test circuits to compute fidelity between skeleton frames
2. **Quantum Subspace Learning**: Variational PCA for dimensionality reduction (optional)
3. **Quantum DTW Path Refinement**: QAOA-based optimization of alignment paths

## 🏗️ Architecture

```
Classical DTW → Extract Window → Quantum Refinement (QAOA) → Refined Path
                     ↓
              Quantum Similarity
              (Swap Test)
```

### Pipeline Components

1. **Data Loading** (`src/data/msr_action3d.py`)
   - Loads MSR Action3D skeleton sequences (20 joints × 3 coords = 60D)
   - Cross-subject split (train: subjects 1,3,5,7,9; test: 2,4,6,8,10)
   - Z-score normalization
   - Interpolation to fixed length

2. **Classical DTW** (`src/dtw/core.py`)
   - Standard DTW with Sakoe-Chiba band constraint
   - 1-NN classifier
   - Baseline for comparison

3. **PCA Projection** (`src/subspace/pca.py`)
   - Reduces 60D → 8D (quantum-friendly dimension)
   - Makes quantum circuits tractable

4. **Quantum Components**:
   - **Amplitude Encoding** (`src/quantum/amplitude_encoding.py`): Prepares quantum states from frames
   - **Swap Test** (`src/quantum/swap_fidelity.py`): Computes quantum similarity F = |⟨ψ|φ⟩|²
   - **QUBO Formulation** (`src/quantum/dtw_qubo.py`): Encodes DTW as binary optimization
   - **QAOA Solver** (`src/quantum/qaoa_solver.py`): Quantum approximate optimization

5. **Path Refinement** (`src/dtw/window_extract.py`)
   - Extracts local windows around classical paths
   - Makes quantum optimization tractable (small problem sizes)

## 📊 Dataset

**MSR Action3D Dataset**
- 20 action classes
- 10 subjects
- 2-3 executions per action-subject pair
- 567 total sequences
- Each frame: 20 joints with (x, y, z) coordinates

**File naming**: `a{action}_s{subject}_e{execution}_skeleton.txt`

Example: `a01_s03_e02_skeleton.txt` = Action 1, Subject 3, Execution 2

## 🚀 Setup

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- numpy
- scipy
- scikit-learn
- qiskit >= 0.45.0
- qiskit-aer
- matplotlib

### Data Preparation

Place the MSR Action3D skeleton files in `msr_action_data/`:

```
msr_action_data/
  a01_s01_e01_skeleton.txt
  a01_s01_e02_skeleton.txt
  ...
```

## 🎮 Usage

### Run Full Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess data
2. Apply PCA projection
3. Run classical DTW baseline
4. Run quantum similarity experiments
5. Run QAOA path refinement
6. Save results to `results/`

### Individual Experiments

```bash
# Classical baseline only
python experiments/e1_baseline.py

# Quantum similarity evaluation
python experiments/e2_quantum_similarity.py

# QAOA path refinement
python experiments/e3_qaoa_refinement.py

# Full classification comparison
python experiments/e4_classification.py
```

### Test Individual Components

```bash
# Test data loading
python src/data/msr_action3d.py

# Test DTW
python src/dtw/core.py

# Test quantum swap test
python src/quantum/swap_fidelity.py

# Test QAOA
python src/quantum/qaoa_solver.py
```

## 📈 Experiments

### E1: Classical DTW Baseline
- Metric: Classification accuracy
- Uses: 1-NN with DTW distance
- Output: `results/classical_dtw_baseline.json`

### E2: Quantum Similarity
- Computes quantum fidelity on DTW paths
- Compares classical vs quantum costs
- Output: `results/quantum_similarity_experiment.json`

### E3: QAOA Path Refinement
- Extracts local windows (12×12 grids)
- Formulates as QUBO
- Solves with QAOA
- Compares refined vs classical paths
- Output: `results/qaoa_refinement_experiment.json`

### E4: Full Classification
- End-to-end classification with quantum components
- Measures accuracy improvement
- Output: `results/classification_comparison.json`

## 🔬 Key Parameters

Configured in `main.py`:

```python
d_q = 8              # Quantum dimension (after PCA)
window = 10          # DTW Sakoe-Chiba band width
shots = 512          # Quantum circuit measurement shots
qaoa_p = 2           # QAOA circuit depth (number of layers)
window_length = 12   # Subsequence length for QAOA
band_width = 2       # Window width for refinement
```

## 📊 Results Structure

Results are saved in JSON format:

```json
{
  "accuracy": 0.85,
  "total_time": 123.45,
  "classical_costs": [...],
  "quantum_costs": [...],
  "improvements": [...],
  "summary": {
    "mean_improvement": 0.12,
    "num_improvements": 8
  }
}
```

## 🧪 Why This is Quantum

This implementation uses **genuine quantum computing**, not quantum-inspired algorithms:

1. **Quantum States**: Prepares actual quantum states |ψ⟩ from skeleton frames using amplitude encoding
2. **Swap Test**: Implements the quantum swap test circuit to measure |⟨ψ|φ⟩|²
3. **QAOA**: Uses Quantum Approximate Optimization Algorithm to solve QUBO
4. **Quantum Simulator**: Runs on Qiskit Aer simulator with realistic quantum noise
5. **Resource Tracking**: Logs qubits, circuit depth, gate counts

### Quantum Resources

For d_q = 8 dimensions:
- **Swap Test**: 2×3 + 1 = 7 qubits (2 registers + 1 ancilla)
- **QAOA**: ~50-100 qubits for 12×12 window
- **Circuit Depth**: O(p × n) where p is QAOA layers, n is problem size

## 📁 Project Structure

```
q-subdtw/
├── main.py                          # Main pipeline orchestrator
├── requirements.txt                 # Dependencies
├── Plan.md                         # Detailed project plan
├── src/
│   ├── data/
│   │   └── msr_action3d.py        # Dataset loader
│   ├── dtw/
│   │   ├── core.py                # Classical DTW
│   │   └── window_extract.py      # Window extraction for refinement
│   ├── subspace/
│   │   └── pca.py                 # PCA projection
│   └── quantum/
│       ├── amplitude_encoding.py   # Quantum state preparation
│       ├── swap_fidelity.py       # Swap test circuit
│       ├── dtw_qubo.py            # QUBO formulation
│       └── qaoa_solver.py         # QAOA implementation
├── experiments/
│   ├── e1_baseline.py
│   ├── e2_quantum_similarity.py
│   ├── e3_qaoa_refinement.py
│   └── e4_classification.py
├── data/
│   └── splits/
│       └── msr_cs.json            # Train/test split info
├── results/                        # Experiment results
└── msr_action_data/               # Skeleton data files
```

## 🔍 Key Features

✅ **Real Quantum Primitives**: Swap test, QAOA, amplitude encoding  
✅ **Hybrid Architecture**: Combines classical and quantum optimally  
✅ **Tractable Problem Sizes**: Uses windowing to make QAOA feasible  
✅ **Reproducible**: Fixed train/test split, random seeds  
✅ **Well-Documented**: Extensive logging and code comments  
✅ **Modular Design**: Each component independently testable  

## 📖 References

1. MSR Action3D Dataset
2. Dynamic Time Warping (DTW)
3. Quantum Swap Test
4. QAOA (Farhi et al., 2014)
5. Qiskit Documentation

## 🎯 Future Extensions

- [ ] Variational Quantum PCA instead of classical PCA
- [ ] Quantum kernel methods for frame similarity
- [ ] VQE for path selection
- [ ] Real quantum hardware experiments (IBMQ)
- [ ] Noise resilience analysis
- [ ] Larger window sizes with problem decomposition

## 🤝 Contributing

This is a research project. Feel free to:
- Test on different datasets
- Experiment with QAOA parameters
- Try different quantum encodings
- Add noise models for realistic simulation

## 📝 License

MIT License

## 👥 Authors

Research project for quantum machine learning and action recognition.

---

**Note**: This implementation runs on a quantum simulator (Qiskit Aer). For real quantum hardware, reduce problem sizes further and add error mitigation.
