"""
QAOA Solver for DTW QUBO
Solves the DTW path selection QUBO using QAOA (Quantum Approximate Optimization Algorithm).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer, AerSimulator
from qiskit.compiler import transpile
from scipy.optimize import minimize
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def qubo_to_ising(qubo: Dict[Tuple[int, int], float]) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Convert QUBO to Ising Hamiltonian.
    
    QUBO uses binary variables x_i ∈ {0, 1}
    Ising uses spin variables s_i ∈ {-1, +1}
    
    Conversion: x_i = (1 + s_i) / 2
    
    Args:
        qubo: QUBO dictionary {(i, j): weight}
        
    Returns:
        (h, J, offset) where:
        - h: linear coefficients {i: h_i}
        - J: interaction coefficients {(i, j): J_ij}
        - offset: constant offset
    """
    h = {}  # Linear terms
    J = {}  # Quadratic terms
    offset = 0.0
    
    for (i, j), weight in qubo.items():
        if i == j:
            # Diagonal term: Q_ii * x_i
            # = Q_ii * (1 + s_i) / 2
            # = Q_ii/2 + Q_ii/2 * s_i
            h[i] = h.get(i, 0) + weight / 2
            offset += weight / 2
        else:
            # Off-diagonal: Q_ij * x_i * x_j
            # = Q_ij * (1 + s_i)(1 + s_j) / 4
            # = Q_ij/4 * (1 + s_i + s_j + s_i*s_j)
            # = Q_ij/4 + Q_ij/4 * s_i + Q_ij/4 * s_j + Q_ij/4 * s_i*s_j
            ii, jj = min(i, j), max(i, j)
            J[(ii, jj)] = J.get((ii, jj), 0) + weight / 4
            h[ii] = h.get(ii, 0) + weight / 4
            h[jj] = h.get(jj, 0) + weight / 4
            offset += weight / 4
    
    return h, J, offset


def create_qaoa_circuit(
    n_qubits: int,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    p: int = 1
) -> QuantumCircuit:
    """
    Create QAOA circuit for Ising Hamiltonian.
    
    QAOA alternates between:
    1. Cost Hamiltonian: exp(-i γ H_C)
    2. Mixer Hamiltonian: exp(-i β H_M) where H_M = Σ X_i
    
    Args:
        n_qubits: Number of qubits
        h: Linear coefficients
        J: Interaction coefficients
        p: Number of QAOA layers
        
    Returns:
        Parameterized QAOA circuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Initialize in superposition
    qc.h(range(n_qubits))
    
    # Create parameters
    gammas = [Parameter(f'γ_{i}') for i in range(p)]
    betas = [Parameter(f'β_{i}') for i in range(p)]
    
    for layer in range(p):
        # Cost Hamiltonian: exp(-i γ H_C)
        # H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j
        
        # Linear terms: h_i Z_i -> RZ(2 * γ * h_i)
        for i, h_val in h.items():
            qc.rz(2 * gammas[layer] * h_val, i)
        
        # Interaction terms: J_ij Z_i Z_j -> CNOT + RZ + CNOT
        for (i, j), J_val in J.items():
            qc.cx(i, j)
            qc.rz(2 * gammas[layer] * J_val, j)
            qc.cx(i, j)
        
        # Mixer Hamiltonian: exp(-i β H_M) where H_M = Σ X_i
        # X_i -> RX(2 * β)
        for i in range(n_qubits):
            qc.rx(2 * betas[layer], i)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc


def evaluate_energy(
    bitstring: str,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float
) -> float:
    """
    Evaluate Ising energy for a bitstring.
    
    E = offset + Σ h_i s_i + Σ J_ij s_i s_j
    
    where s_i = +1 if bit is 0, -1 if bit is 1
    (We use the convention: |0⟩ -> s=+1, |1⟩ -> s=-1)
    
    Args:
        bitstring: Binary string (e.g., "01101")
        h: Linear coefficients
        J: Interaction coefficients
        offset: Constant offset
        
    Returns:
        Energy value
    """
    n = len(bitstring)
    spins = [1 if bit == '0' else -1 for bit in bitstring]
    
    energy = offset
    
    # Linear terms
    for i, h_val in h.items():
        if i < n:
            energy += h_val * spins[i]
    
    # Interaction terms
    for (i, j), J_val in J.items():
        if i < n and j < n:
            energy += J_val * spins[i] * spins[j]
    
    return energy


class QAOASolver:
    """
    QAOA solver for QUBO problems.
    """
    
    def __init__(
        self,
        p: int = 2,
        shots: int = 1024,
        backend_name: str = 'qasm_simulator',
        optimizer: str = 'COBYLA',
        maxiter: int = 100
    ):
        """
        Initialize QAOA solver.
        
        Args:
            p: Number of QAOA layers
            shots: Number of measurement shots
            backend_name: Qiskit backend
            optimizer: Classical optimizer ('COBYLA', 'SLSQP', etc.)
            maxiter: Maximum optimization iterations
        """
        self.p = p
        self.shots = shots
        self.backend = Aer.get_backend(backend_name)
        self.optimizer = optimizer
        self.maxiter = maxiter
        
        logger.info(f"QAOASolver initialized: p={p}, shots={shots}, optimizer={optimizer}")
    
    def solve(
        self,
        qubo: Dict[Tuple[int, int], float],
        n_qubits: int = None
    ) -> Tuple[str, float, dict]:
        """
        Solve QUBO problem using QAOA.
        
        Args:
            qubo: QUBO dictionary
            n_qubits: Number of qubits (if None, inferred from QUBO)
            
        Returns:
            (best_bitstring, best_energy, results)
        """
        # Infer number of qubits
        if n_qubits is None:
            max_var = max(max(i, j) for i, j in qubo.keys())
            n_qubits = max_var + 1
        
        logger.info(f"Solving QUBO with {n_qubits} qubits, {len(qubo)} terms")
        
        # Convert QUBO to Ising
        h, J, offset = qubo_to_ising(qubo)
        
        logger.info(f"Ising: {len(h)} linear terms, {len(J)} interactions, offset={offset:.4f}")
        
        # Create QAOA circuit
        qaoa_circuit = create_qaoa_circuit(n_qubits, h, J, p=self.p)
        
        logger.info(f"QAOA circuit: depth={qaoa_circuit.depth()}, "
                   f"params={len(qaoa_circuit.parameters)}")
        
        # Objective function for classical optimization
        def objective(params):
            # Bind parameters
            param_dict = {}
            for i in range(self.p):
                param_dict[f'γ_{i}'] = params[i]
                param_dict[f'β_{i}'] = params[self.p + i]
            
            bound_circuit = qaoa_circuit.bind_parameters(param_dict)
            
            # Transpile and run
            transpiled = transpile(bound_circuit, self.backend)
            job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Compute expectation value
            expectation = 0.0
            for bitstring, count in counts.items():
                energy = evaluate_energy(bitstring, h, J, offset)
                expectation += energy * count / self.shots
            
            return expectation
        
        # Initial parameters
        initial_params = np.random.uniform(0, 2*np.pi, 2*self.p)
        
        logger.info("Starting classical optimization...")
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            method=self.optimizer,
            options={'maxiter': self.maxiter}
        )
        
        logger.info(f"Optimization completed: success={result.success}, "
                   f"iterations={result.nit}, energy={result.fun:.4f}")
        
        # Get best solution
        optimal_params = result.x
        param_dict = {}
        for i in range(self.p):
            param_dict[f'γ_{i}'] = optimal_params[i]
            param_dict[f'β_{i}'] = optimal_params[self.p + i]
        
        bound_circuit = qaoa_circuit.bind_parameters(param_dict)
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        final_result = job.result()
        counts = final_result.get_counts()
        
        # Find lowest energy bitstring
        best_bitstring = None
        best_energy = float('inf')
        
        for bitstring, count in counts.items():
            energy = evaluate_energy(bitstring, h, J, offset)
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
        
        results = {
            'optimal_params': optimal_params,
            'counts': counts,
            'optimization_result': result,
            'circuit_depth': qaoa_circuit.depth(),
            'num_qubits': n_qubits
        }
        
        return best_bitstring, best_energy, results


if __name__ == "__main__":
    # Test QAOA on a simple QUBO
    
    # Max-Cut example on 4-node graph
    # Nodes: 0, 1, 2, 3
    # Edges: (0,1), (1,2), (2,3), (3,0)
    # QUBO: maximize sum of cut edges
    # Each edge (i,j) contributes: -x_i*x_j - x_i + x_j + 1
    # to favor x_i ≠ x_j
    
    qubo = {
        (0, 0): -1,
        (1, 1): -1,
        (2, 2): -1,
        (3, 3): -1,
        (0, 1): 2,
        (1, 2): 2,
        (2, 3): 2,
        (0, 3): 2
    }
    
    print("Testing QAOA on Max-Cut QUBO")
    print(f"QUBO terms: {qubo}")
    
    solver = QAOASolver(p=2, shots=1024, maxiter=50)
    best_solution, best_energy, results = solver.solve(qubo, n_qubits=4)
    
    print(f"\nBest solution: {best_solution}")
    print(f"Best energy: {best_energy:.4f}")
    print(f"Circuit depth: {results['circuit_depth']}")
    print(f"Top 5 solutions:")
    for bitstring, count in sorted(results['counts'].items(), key=lambda x: -x[1])[:5]:
        h, J, offset = qubo_to_ising(qubo)
        energy = evaluate_energy(bitstring, h, J, offset)
        print(f"  {bitstring}: count={count}, energy={energy:.4f}")
