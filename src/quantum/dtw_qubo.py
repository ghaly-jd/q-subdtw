"""
QUBO Formulation for DTW Path Selection
Formulates the DTW alignment problem as a QUBO for quantum optimization.
"""

import numpy as np
from typing import Set, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def build_dtw_qubo(
    window_cells: Set[Tuple[int, int]],
    costs: Dict[Tuple[int, int], float],
    T_q: int,
    T_c: int,
    penalty_weight: float = 10.0
) -> Tuple[Dict[Tuple[Tuple[int, int], Tuple[int, int]], float], Dict[Tuple[int, int], int]]:
    """
    Build QUBO formulation for DTW path selection in a window.
    
    We formulate path selection as a binary optimization problem:
    - Binary variable x_{i,j} = 1 if cell (i,j) is on the path, 0 otherwise
    
    Objective:
        H = H_cost + λ * H_constraint
    
    where:
    - H_cost = Σ c_{i,j} * x_{i,j}  (minimize alignment cost)
    - H_constraint = penalties for invalid paths
    
    Constraints (enforced as soft penalties):
    1. Start at (1,1): x_{1,1} = 1
    2. End at (T_q, T_c): x_{T_q,T_c} = 1
    3. Monotonicity: if x_{i,j} = 1, then x_{i',j'} = 1 for some predecessor
    4. Path connectivity: cells form a connected path
    
    Args:
        window_cells: Set of valid (i,j) cells in the window
        costs: Dict mapping (i,j) -> local cost
        T_q: Query subsequence length
        T_c: Candidate subsequence length
        penalty_weight: Weight for constraint violations (λ)
        
    Returns:
        (qubo_dict, var_mapping) where:
        - qubo_dict: QUBO coefficients as {(var1, var2): weight}
        - var_mapping: Maps (i,j) cells to variable indices
    """
    # Create variable mapping: (i,j) -> variable index
    var_mapping = {}
    reverse_mapping = {}
    idx = 0
    for cell in sorted(window_cells):
        var_mapping[cell] = idx
        reverse_mapping[idx] = cell
        idx += 1
    
    n_vars = len(var_mapping)
    logger.info(f"Building QUBO with {n_vars} variables")
    
    # Initialize QUBO dictionary
    qubo = {}
    
    # 1. Cost terms: H_cost = Σ c_{i,j} * x_{i,j}
    # This is linear, so it goes on the diagonal
    for cell, cost in costs.items():
        if cell not in var_mapping:
            continue
        var_idx = var_mapping[cell]
        qubo[(var_idx, var_idx)] = qubo.get((var_idx, var_idx), 0) + cost
    
    # 2. Start constraint: penalize if NOT at start
    # We want x_{1,1} = 1, so add penalty for x_{1,1} = 0
    # Penalty: λ * (1 - x_{1,1})² = λ * (1 - 2*x_{1,1} + x_{1,1}²)
    #                              = λ * (1 - x_{1,1})  [since x² = x for binary]
    if (1, 1) in var_mapping:
        var_start = var_mapping[(1, 1)]
        # Add negative coefficient to encourage x_{1,1} = 1
        qubo[(var_start, var_start)] = qubo.get((var_start, var_start), 0) - penalty_weight
    
    # 3. End constraint: similar to start
    if (T_q, T_c) in var_mapping:
        var_end = var_mapping[(T_q, T_c)]
        qubo[(var_end, var_end)] = qubo.get((var_end, var_end), 0) - penalty_weight
    
    # 4. Path connectivity: each cell (except start) should have a predecessor
    # For each cell (i,j), penalize if it's selected but no valid predecessor is selected
    # Valid predecessors of (i,j): (i-1,j), (i,j-1), (i-1,j-1)
    #
    # Penalty: if x_{i,j} = 1 but all predecessors are 0
    # This is: x_{i,j} * (1 - x_{i-1,j}) * (1 - x_{i,j-1}) * (1 - x_{i-1,j-1})
    #
    # For QUBO, we use a simpler formulation:
    # Encourage: x_{i,j} => at least one predecessor
    # Penalty: λ * x_{i,j} * (1 - sum_of_predecessors)
    #
    # Simplified: λ * (x_{i,j} - x_{i,j}*x_{pred1} - x_{i,j}*x_{pred2} - x_{i,j}*x_{pred3})
    
    for cell in window_cells:
        i, j = cell
        if i == 1 and j == 1:  # Skip start cell
            continue
        
        if cell not in var_mapping:
            continue
        
        var_current = var_mapping[cell]
        
        # Find valid predecessors
        predecessors = []
        for pred_cell in [(i-1, j), (i, j-1), (i-1, j-1)]:
            if pred_cell in var_mapping:
                predecessors.append(var_mapping[pred_cell])
        
        if not predecessors:
            continue
        
        # Add penalty term: encourage at least one predecessor
        # For simplicity, we add: λ * x_{i,j} - λ * Σ x_{i,j} * x_pred
        qubo[(var_current, var_current)] = qubo.get((var_current, var_current), 0) + penalty_weight
        
        for var_pred in predecessors:
            # Negative coefficient for x_current * x_pred (encourages both to be 1)
            key = tuple(sorted([var_current, var_pred]))
            qubo[key] = qubo.get(key, 0) - penalty_weight / len(predecessors)
    
    # 5. Successor constraint (optional, for stronger connectivity)
    # Each cell (except end) should have a successor
    for cell in window_cells:
        i, j = cell
        if i == T_q and j == T_c:  # Skip end cell
            continue
        
        if cell not in var_mapping:
            continue
        
        var_current = var_mapping[cell]
        
        # Find valid successors
        successors = []
        for succ_cell in [(i+1, j), (i, j+1), (i+1, j+1)]:
            if succ_cell in var_mapping:
                successors.append(var_mapping[succ_cell])
        
        if not successors:
            continue
        
        # Similar penalty as predecessors
        qubo[(var_current, var_current)] = qubo.get((var_current, var_current), 0) + penalty_weight
        
        for var_succ in successors:
            key = tuple(sorted([var_current, var_succ]))
            qubo[key] = qubo.get(key, 0) - penalty_weight / len(successors)
    
    logger.info(f"QUBO built with {len(qubo)} terms")
    
    return qubo, var_mapping


def decode_qubo_solution(
    bitstring: str,
    var_mapping: Dict[Tuple[int, int], int]
) -> List[Tuple[int, int]]:
    """
    Decode QUBO solution (bitstring) back to DTW path.
    
    Args:
        bitstring: Binary solution string (e.g., "0110101...")
        var_mapping: Maps (i,j) cells to variable indices
        
    Returns:
        List of (i,j) cells selected in the solution
    """
    # Reverse mapping: var_idx -> (i,j)
    reverse_mapping = {v: k for k, v in var_mapping.items()}
    
    selected_cells = []
    for var_idx, bit in enumerate(bitstring):
        if bit == '1' and var_idx in reverse_mapping:
            selected_cells.append(reverse_mapping[var_idx])
    
    # Sort by (i, j) to form a path
    selected_cells.sort()
    
    return selected_cells


def validate_path(
    path: List[Tuple[int, int]],
    T_q: int,
    T_c: int
) -> Tuple[bool, str]:
    """
    Validate if a path is a valid DTW path.
    
    Valid DTW path must:
    1. Start at (1, 1)
    2. End at (T_q, T_c)
    3. Be monotonic (i and j never decrease)
    4. Only move to adjacent cells: (i+1,j), (i,j+1), or (i+1,j+1)
    
    Args:
        path: List of (i,j) coordinates
        T_q: Query length
        T_c: Candidate length
        
    Returns:
        (is_valid, message)
    """
    if not path:
        return False, "Empty path"
    
    # Check start
    if path[0] != (1, 1):
        return False, f"Path doesn't start at (1,1): starts at {path[0]}"
    
    # Check end
    if path[-1] != (T_q, T_c):
        return False, f"Path doesn't end at ({T_q},{T_c}): ends at {path[-1]}"
    
    # Check monotonicity and adjacency
    for k in range(len(path) - 1):
        i1, j1 = path[k]
        i2, j2 = path[k+1]
        
        # Check monotonicity
        if i2 < i1 or j2 < j1:
            return False, f"Non-monotonic transition: ({i1},{j1}) -> ({i2},{j2})"
        
        # Check adjacency
        di = i2 - i1
        dj = j2 - j1
        if di > 1 or dj > 1 or (di == 0 and dj == 0):
            return False, f"Invalid transition: ({i1},{j1}) -> ({i2},{j2})"
    
    return True, "Valid path"


def evaluate_path_cost(
    path: List[Tuple[int, int]],
    costs: Dict[Tuple[int, int], float]
) -> float:
    """
    Evaluate the total cost of a path.
    
    Args:
        path: List of (i,j) coordinates
        costs: Dict mapping (i,j) -> cost
        
    Returns:
        Total path cost
    """
    total_cost = 0.0
    for cell in path:
        if cell in costs:
            total_cost += costs[cell]
        else:
            logger.warning(f"Cell {cell} not in cost dict")
    
    return total_cost


if __name__ == "__main__":
    # Test QUBO building
    
    # Create a small window
    T_q = 5
    T_c = 5
    window_cells = {(i, j) for i in range(1, T_q+1) for j in range(1, T_c+1) 
                    if abs(i-j) <= 1}  # Diagonal band
    
    # Create simple costs
    costs = {cell: np.random.rand() for cell in window_cells}
    
    print(f"Window cells: {len(window_cells)}")
    print(f"Costs: {len(costs)}")
    
    # Build QUBO
    qubo, var_mapping = build_dtw_qubo(
        window_cells, costs, T_q, T_c, penalty_weight=5.0
    )
    
    print(f"\nQUBO terms: {len(qubo)}")
    print(f"Variables: {len(var_mapping)}")
    print(f"Variable mapping sample: {list(var_mapping.items())[:5]}")
    
    # Test solution decoding
    print("\n" + "="*50)
    
    # Create a valid path
    path = [(i, i) for i in range(1, 6)]
    print(f"Test path: {path}")
    
    # Create bitstring (all zeros except path cells)
    bitstring = ['0'] * len(var_mapping)
    for cell in path:
        if cell in var_mapping:
            bitstring[var_mapping[cell]] = '1'
    bitstring = ''.join(bitstring)
    
    # Decode
    decoded_path = decode_qubo_solution(bitstring, var_mapping)
    print(f"Decoded path: {decoded_path}")
    
    # Validate
    is_valid, msg = validate_path(decoded_path, T_q, T_c)
    print(f"Valid: {is_valid}, Message: {msg}")
    
    # Evaluate cost
    cost = evaluate_path_cost(decoded_path, costs)
    print(f"Path cost: {cost:.4f}")
