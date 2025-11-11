
# Quantum DTW for Skeleton Actions (Q-DTW-MSRA)
**Goal:** build a hybrid pipeline where *some steps are strictly quantum (or variational on a simulator)* — not only “quantum-inspired.”  
**Data:** MSR Action3D (60-D sequences, fixed cross-subject split)

---

## 1. Idea in 30 seconds

Dynamic Time Warping (DTW) = choose a low-cost path in an alignment grid between two sequences.

We’ll make it quantum in **three** places:

1. **Quantum frame similarity**: use a swap test / quantum kernel to get the local cost between 2 frames (small dimension, real quantum primitive).
2. **Quantum subspace**: learn a low-dim basis with a **variational PCA / qPCA-style circuit** instead of pure numpy.
3. **Quantum DTW path refinement**: take the (classical) DTW cost matrix for two sequences, **extract a small window** around the classical path, and **run QAOA on that window** to pick the optimal admissible path. That’s a real quantum optimization over a DTW-like objective.

So the story becomes:
> “We do classical DTW to get a candidate, but the final alignment (in a local neighborhood) is chosen by a quantum optimizer that sees quantum-similarity costs.”

That’s a real hybrid pipeline.

---

## 2. Pipeline overview

1. **Data & preprocessing (classical)**  
   - load MSR Action3D  
   - interpolate, z-score  
   - fix cross-subject split

2. **Dimensionality control (hybrid)**  
   - do a *classical* PCA to go from 60 → d_small (like 8–12) **only to make quantum cheaper**  
   - alternatively, train a *variational quantum PCA* on mini-batches to get the same d_small

3. **Local cost (quantum)**  
   - for frame pairs inside that small d_small space, run a **swap test** (or a quantum kernel circuit) → get similarity  
   - cost = 1 − similarity  
   - fill a **quantum-aware cost matrix** for a small region

4. **Global DTW (classical)**  
   - run normal DTW on the (possibly classical) cost matrix → get a path
   - this gives us a good but not necessarily optimal path

5. **Path refinement (quantum optimization)**  
   - take a band around the classical path (width r) → this is a small grid  
   - encode “valid DTW path” as a QUBO / Ising  
   - run QAOA / VQE on that small problem  
   - pick lowest-energy solution → **this is the quantum-refined DTW path**

6. **Evaluation**  
   - classification = 1-NN using refined DTW distances  
   - compare: classical DTW vs quantum-similarity DTW vs quantum-refined DTW  
   - report runtime and qubits

---

## 3. Data & preprocessing

**File:** `src/data/msr_action3d.py`

- Returns:
  - `train_sequences: List[np.ndarray[T_i, 60]]`
  - `train_labels: List[int]`
  - `test_sequences, test_labels`
- Steps:
  1. load skeletons
  2. drop bad frames
  3. z-score using train mean/std
  4. save split to `data/splits/msr_cs.json`

**Why:** reproducible split = fair comparison between classical DTW and hybrid quantum DTW.

---

## 4. Classical DTW baseline

**File:** `src/dtw/core.py`

- Local cost (Euclidean):
  \[
  \delta(q_i, c_j) = \| q_i - c_j \|_2
  \]
- Recurrence:
  \[
  D_{i,j} = \delta(q_i, c_j) + \min(D_{i-1,j}, D_{i,j-1}, D_{i-1,j-1})
  \]
  with window \(|i-j| \le w\).

- 1-NN classifier:
  - for test seq Q, compute DTW(Q, train_k) ∀k
  - choose train label of min distance

We store: accuracy, avg DTW time, path length.

This is **baseline 0**.

---

## 5. Make it quantum-friendly: reduce dimension first

Right now frames are 60-D → too big for nice, cheap quantum circuits.

We define a **target quantum dimension** \(d_q\) (e.g. 8 or 12). Our quantum routines will only ever see \(d_q\), not 60.

Two ways to get it:

### 5.1 Classical PCA (fast path)
**File:** `src/subspace/pca.py`
- stack sampled frames  
- PCA → take first \(d_q\) components  
- project every frame to \(\mathbb{R}^{d_q}\)

### 5.2 Variational Quantum PCA (quantum path)
**File:** `src/quantum/vq_pca.py`
- define a parameterized circuit \(U(\theta)\) on \(\log_2 d_q\) qubits
- objective: maximize variance of embedded data (or minimize reconstruction error)
- optimize \(\theta\) with classical optimizer
- resulting circuit is the “encoder”

**Why:** now we can say “the representation used by DTW comes from a quantum variational model,” not just numpy.

---

## 6. Quantum frame similarity (real quantum primitive)

This is the first **truly quantum** component.

**File:** `src/quantum/swap_fidelity.py`

### 6.1 Amplitude/state preparation
Given two d_q-dimensional *real* vectors:
\[
x, y \in \mathbb{R}^{d_q}
\]
normalize them:
\[
\tilde{x} = \frac{x}{\|x\|}, \quad \tilde{y} = \frac{y}{\|y\|}
\]
prepare states \(|x\rangle, |y\rangle\) on \(n = \lceil \log_2 d_q \rceil\) qubits.

### 6.2 Swap test
Standard swap test circuit gives:
\[
P(\text{ancilla}=0) = \frac{1 + |\langle x | y \rangle|^2}{2}
\]
so
\[
F(x,y) = |\langle x | y \rangle|^2 = 2 P(0) - 1
\]

Define **quantum local cost**:
\[
\delta_Q(x,y) = 1 - F(x,y)
\]

We will **not** do this for every (i, j) pair (too many). Instead:

- we run it for pairs **inside a band** around the classical DTW path
- or for the **top-K most promising** alignments
- or for a **reduced-length version** of the sequences

This way, we really call a quantum primitive, and we use its output in DTW.

---

## 7. Quantum DTW path refinement (the actually-new part)

This is the part that makes the work stand out.

### 7.1 DTW as a path problem
Classical DTW is finding a monotone path from (1,1) to (n,m) on a grid, minimizing the sum of local costs.

We can view this as:  
“select cells in the grid that form a valid DTW path and minimize total cost.”

That’s a combinatorial optimization → we can rewrite a **small** version as QUBO.

### 7.2 Why “small”?
A full 100×100 grid → 10,000 variables → too big.  
But if we take a **band of width r** around the classical path (say r=3 or r=5), we get maybe 200–500 cells → that’s something we can downscale even more (subsequences of length 20–30) and send to a quantum optimizer.

So the procedure is:

1. Run classical DTW → get path P
2. Extract subsequences Q[ t1:t2 ], C[ s1:s2 ] of length L (e.g. L = 24)
3. Build alignment grid for those subsequences only
4. Only keep a band of width r around the classical mini-path
5. Formulate QUBO on this tiny grid
6. Run QAOA

### 7.3 QUBO formulation (sketch)
Let \(x_{i,j} \in \{0,1\}\) be “we visit cell (i,j).”

**Cost term:**
\[
H_{\text{cost}} = \sum_{i,j} c_{i,j} x_{i,j}
\]
where \(c_{i,j}\) comes from **quantum local cost** \(\delta_Q\) above.

**Path constraints:**
1. start at (1,1): \(x_{1,1} = 1\)  
2. end at (L,L) (or appropriate): \(x_{L,L} = 1\)
3. monotonicity: from (i,j) you can only go to (i+1,j), (i,j+1), (i+1,j+1)

We enforce this softly:
\[
H_{\text{cons}} = \lambda \sum_{i,j} \left( x_{i,j} - (x_{i-1,j} + x_{i,j-1} + x_{i-1,j-1}) \right)_+ 
\]
In practice we build it as a binary penalty (typical QUBO style): add penalties when a cell is on but none of its parents are on, etc.

**Total Hamiltonian:**
\[
H = H_{\text{cost}} + H_{\text{cons}}
\]

We feed H to QAOA (on simulator).

### 7.4 Outcome
- QAOA returns a bitstring → which cells to pick
- decode into a path
- compare its cost to classical DTW on that same window

If the quantum-refined path has lower (quantum) cost, we just proved the quantum optimizer actually contributed.

---

## 8. Hybrid subspaces (still useful)

We keep the “global + per-class small subspace” idea, but now we can say:

- global subspace can be learned classically  
- per-class small subspace can be tried as a **small variational quantum circuit** (because its size is tiny)

This gives you one more place where quantum shows up.

---

## 9. Experiments to run

1. **E1: Classical DTW baseline**  
   - metric: accuracy, runtime

2. **E2: Quantum-similarity DTW (partial)**  
   - compute classical DTW  
   - re-evaluate local costs on the DTW path using swap test  
   - recompute path cost  
   - report difference

3. **E3: Quantum DTW path refinement**  
   - pick 50 random test pairs  
   - for each: classical DTW → extract window → QAOA → compare costs  
   - report how many times QAOA found a better (quantum-cost) path

4. **E4: Classification with refined distances**  
   - for test seq, pick nearest neighbor using *refined* distance  
   - compare accuracy

5. **E5: Ablation on dimension**  
   - d_q ∈ {4, 8, 12}  
   - report qubits, circuit depth

6. **E6: Runtime/resource table**  
   - #qubits for swap test  
   - #qubits for QAOA window  
   - circuit depth

---

## 10. Why this is actually quantum

- We **prepare quantum states** of skeleton frames (after projection).
- We **measure similarities** via swap test / quantum kernel — a known quantum primitive.
- We **formulate part of DTW as QUBO** and solve it with **QAOA** — that’s a genuine quantum optimization algorithm.
- We **compare** to the classical version on the same data.
- We **log resources** so it’s clear how to move to hardware.

That’s enough to say: *“This is a quantum approach to DTW on skeleton actions, evaluated on MSR.”*

---

## 11. Milestones (again, but quantum)

**Week 1**  
- dataset, classical DTW baseline

**Week 2**  
- PCA → d_q  
- project all sequences

**Week 3**  
- quantum swap test function (on simulator)  
- run it on DTW-path pairs

**Week 4**  
- QUBO builder for mini alignment grids  
- QAOA solver

**Week 5**  
- integrate: classical DTW → quantum refinement  
- measure classification

**Week 6**  
- writeup + plots + resource table

---

## 12. Files to implement

- `src/data/msr_action3d.py`
- `src/dtw/core.py`
- `src/dtw/window_extract.py` (to get local window around path)
- `src/subspace/pca.py`
- `src/quantum/amplitude_encoding.py`
- `src/quantum/swap_fidelity.py`
- `src/quantum/dtw_qubo.py` (builds QUBO from window)
- `src/quantum/qaoa_solver.py`
- `experiments/*.json`

---

## 13. Future extension (optional but sexy)

- replace swap test with **kernel-based quantum SVM** for frame classification
- replace QAOA with **VQE for path selection**
- map to a real backend (small sequences) and report noise effect

---
