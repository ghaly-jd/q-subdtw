"""
Window Extraction from DTW Path
Extracts a local band/window around the classical DTW path for quantum refinement.
"""

import numpy as np
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def extract_window_from_path(
    path: List[Tuple[int, int]],
    band_width: int = 3,
    T_q: int = None,
    T_c: int = None
) -> Set[Tuple[int, int]]:
    """
    Extract a window (band) of cells around a DTW path.
    
    Given a path P = [(i₁,j₁), (i₂,j₂), ..., (iₖ,jₖ)],
    we create a band of width r around each point on the path.
    
    For each (i,j) in path, include all cells (i',j') where:
        max(|i'-i|, |j'-j|) ≤ band_width
    
    Args:
        path: DTW path as list of (i, j) coordinates
        band_width: Width of the band (r)
        T_q: Query sequence length (for boundary checking)
        T_c: Candidate sequence length (for boundary checking)
        
    Returns:
        Set of (i, j) cell coordinates in the window
    """
    window = set()
    
    for i, j in path:
        # Add cells in neighborhood
        for di in range(-band_width, band_width + 1):
            for dj in range(-band_width, band_width + 1):
                i_new = i + di
                j_new = j + dj
                
                # Check boundaries
                if T_q is not None and (i_new < 1 or i_new > T_q):
                    continue
                if T_c is not None and (j_new < 1 or j_new > T_c):
                    continue
                
                window.add((i_new, j_new))
    
    return window


def extract_subsequence_window(
    query: np.ndarray,
    candidate: np.ndarray,
    path: List[Tuple[int, int]],
    window_length: int = 20,
    band_width: int = 3,
    center_idx: int = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Set[Tuple[int, int]], Tuple[int, int]]:
    """
    Extract a local subsequence window for quantum refinement.
    
    This is the key function for making quantum optimization tractable.
    Instead of optimizing the full DTW path, we:
    1. Extract a short subsequence from both query and candidate
    2. Build a small alignment grid for these subsequences
    3. Extract a band around the classical mini-path
    4. Send this small problem to QAOA
    
    Args:
        query: Full query sequence (T_q, d)
        candidate: Full candidate sequence (T_c, d)
        path: Full DTW path
        window_length: Length of subsequences to extract (L)
        band_width: Width of band around mini-path
        center_idx: Index in path to center the window (if None, use middle)
        
    Returns:
        (sub_query, sub_candidate, mini_path, window_cells, offsets)
        where:
        - sub_query: Subsequence from query (L, d)
        - sub_candidate: Subsequence from candidate (L, d)
        - mini_path: Path segment within subsequences
        - window_cells: Set of cells in the refinement window
        - offsets: (i_offset, j_offset) - starting indices in original sequences
    """
    T_q = len(query)
    T_c = len(candidate)
    path_len = len(path)
    
    # Choose center point in path
    if center_idx is None:
        center_idx = path_len // 2
    
    center_i, center_j = path[center_idx]
    
    # Calculate extraction range
    half_len = window_length // 2
    
    # Query range
    i_start = max(1, center_i - half_len)
    i_end = min(T_q, center_i + half_len)
    i_len = i_end - i_start + 1
    
    # Adjust if we hit boundaries
    if i_len < window_length:
        if i_start == 1:
            i_end = min(T_q, i_start + window_length - 1)
        else:
            i_start = max(1, i_end - window_length + 1)
    
    # Candidate range
    j_start = max(1, center_j - half_len)
    j_end = min(T_c, center_j + half_len)
    j_len = j_end - j_start + 1
    
    # Adjust if we hit boundaries
    if j_len < window_length:
        if j_start == 1:
            j_end = min(T_c, j_start + window_length - 1)
        else:
            j_start = max(1, j_end - window_length + 1)
    
    # Extract subsequences (note: path uses 1-indexed, arrays are 0-indexed)
    sub_query = query[i_start-1:i_end, :]
    sub_candidate = candidate[j_start-1:j_end, :]
    
    # Extract mini-path (shift to local coordinates)
    mini_path = []
    for i, j in path:
        if i_start <= i <= i_end and j_start <= j <= j_end:
            local_i = i - i_start + 1  # Convert to 1-indexed local coords
            local_j = j - j_start + 1
            mini_path.append((local_i, local_j))
    
    # Get window cells around mini-path
    L_q = len(sub_query)
    L_c = len(sub_candidate)
    window_cells = extract_window_from_path(
        mini_path,
        band_width=band_width,
        T_q=L_q,
        T_c=L_c
    )
    
    offsets = (i_start, j_start)
    
    logger.info(f"Extracted window: query[{i_start}:{i_end}], candidate[{j_start}:{j_end}]")
    logger.info(f"  Subsequence shapes: {sub_query.shape}, {sub_candidate.shape}")
    logger.info(f"  Mini-path length: {len(mini_path)}")
    logger.info(f"  Window cells: {len(window_cells)}")
    
    return sub_query, sub_candidate, mini_path, window_cells, offsets


def visualize_window(
    window_cells: Set[Tuple[int, int]],
    path: List[Tuple[int, int]],
    T_q: int,
    T_c: int
) -> np.ndarray:
    """
    Create a visualization matrix of the window and path.
    
    Args:
        window_cells: Set of cells in window
        path: DTW path
        T_q: Query length
        T_c: Candidate length
        
    Returns:
        Matrix where:
        - 0: outside window
        - 1: in window but not on path
        - 2: on path
    """
    grid = np.zeros((T_q, T_c))
    
    # Mark window cells
    for i, j in window_cells:
        if 1 <= i <= T_q and 1 <= j <= T_c:
            grid[i-1, j-1] = 1
    
    # Mark path
    for i, j in path:
        if 1 <= i <= T_q and 1 <= j <= T_c:
            grid[i-1, j-1] = 2
    
    return grid


def compute_local_cost_matrix(
    sub_query: np.ndarray,
    sub_candidate: np.ndarray,
    window_cells: Set[Tuple[int, int]],
    distance_fn=None
) -> dict:
    """
    Compute local costs only for cells in the window.
    
    This is used for quantum refinement - we only compute costs
    for the cells we're actually optimizing over.
    
    Args:
        sub_query: Subsequence from query (L_q, d)
        sub_candidate: Subsequence from candidate (L_c, d)
        window_cells: Set of (i, j) cells to compute costs for
        distance_fn: Distance function (if None, use Euclidean)
        
    Returns:
        Dictionary mapping (i, j) -> cost
    """
    if distance_fn is None:
        distance_fn = lambda x, y: np.linalg.norm(x - y)
    
    costs = {}
    
    for i, j in window_cells:
        # Convert to 0-indexed
        cost = distance_fn(sub_query[i-1], sub_candidate[j-1])
        costs[(i, j)] = cost
    
    return costs


if __name__ == "__main__":
    # Test window extraction
    
    # Create a simple path
    path = [(i, i) for i in range(1, 21)]  # Diagonal path from (1,1) to (20,20)
    
    # Extract window
    window = extract_window_from_path(path, band_width=2, T_q=20, T_c=20)
    
    print(f"Path length: {len(path)}")
    print(f"Window size: {len(window)}")
    print(f"First few window cells: {sorted(list(window))[:10]}")
    
    # Test subsequence extraction
    print("\n" + "="*50)
    
    np.random.seed(42)
    query = np.random.randn(50, 8)
    candidate = np.random.randn(50, 8)
    full_path = [(i, i) for i in range(1, 51)]
    
    sub_q, sub_c, mini_path, window_cells, offsets = extract_subsequence_window(
        query, candidate, full_path,
        window_length=15,
        band_width=3
    )
    
    print(f"\nOffsets: {offsets}")
    print(f"Subsequence shapes: {sub_q.shape}, {sub_c.shape}")
    print(f"Mini-path: {mini_path[:5]}...")
    
    # Compute costs
    costs = compute_local_cost_matrix(sub_q, sub_c, window_cells)
    print(f"Computed {len(costs)} costs")
