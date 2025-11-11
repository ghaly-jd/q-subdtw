"""
MSR Action3D Dataset Loader
Loads and preprocesses skeleton sequences from MSR Action3D dataset.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MSRAction3D:
    """
    MSR Action3D dataset loader.
    
    Dataset structure:
    - 20 actions (a01-a20)
    - 10 subjects (s01-s10)
    - 2-3 executions per action-subject pair
    - Each skeleton has 20 joints with (x, y, z) coordinates
    - Frame dimension: 20 joints × 3 coords = 60D
    """
    
    # Cross-subject split as per standard protocol
    # Train: subjects 1, 3, 5, 7, 9
    # Test: subjects 2, 4, 6, 8, 10
    TRAIN_SUBJECTS = [1, 3, 5, 7, 9]
    TEST_SUBJECTS = [2, 4, 6, 8, 10]
    
    NUM_JOINTS = 20
    NUM_COORDS = 3
    FRAME_DIM = NUM_JOINTS * NUM_COORDS  # 60D
    
    def __init__(self, data_dir: str):
        """
        Initialize the MSR Action3D loader.
        
        Args:
            data_dir: Path to directory containing skeleton .txt files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.train_sequences = []
        self.train_labels = []
        self.test_sequences = []
        self.test_labels = []
        
        self.train_mean = None
        self.train_std = None
        
    def parse_skeleton_file(self, filepath: Path) -> np.ndarray:
        """
        Parse a skeleton file and return sequence as numpy array.
        
        Each line has 4 values per joint: x, y, z, confidence
        We only use x, y, z (ignore confidence)
        
        Args:
            filepath: Path to skeleton file
            
        Returns:
            numpy array of shape (num_frames, 60)
        """
        try:
            data = np.loadtxt(filepath)
            
            # Check if data is valid
            if data.size == 0:
                logger.warning(f"Empty file: {filepath}")
                return None
            
            # Reshape: each row is a joint (x, y, z, conf)
            # We take only first 3 columns (x, y, z)
            if data.ndim == 1:
                # Single frame
                data = data.reshape(1, -1)
            
            # Extract x, y, z (drop confidence column)
            # Data format: each row is one joint with 4 values
            num_rows = data.shape[0]
            num_frames = num_rows // self.NUM_JOINTS
            
            if num_rows % self.NUM_JOINTS != 0:
                logger.warning(f"Incomplete frames in {filepath}: {num_rows} rows")
                # Trim to complete frames
                num_frames = num_rows // self.NUM_JOINTS
                data = data[:num_frames * self.NUM_JOINTS]
            
            # Reshape to (num_frames, num_joints, 4) then extract x,y,z
            frames = data.reshape(num_frames, self.NUM_JOINTS, 4)[:, :, :3]
            
            # Flatten to (num_frames, 60)
            frames_flat = frames.reshape(num_frames, self.FRAME_DIM)
            
            # Remove frames with all zeros (bad frames)
            valid_frames = ~np.all(frames_flat == 0, axis=1)
            frames_flat = frames_flat[valid_frames]
            
            if len(frames_flat) == 0:
                logger.warning(f"No valid frames in {filepath}")
                return None
                
            return frames_flat
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def extract_metadata(self, filename: str) -> Tuple[int, int, int]:
        """
        Extract action, subject, execution from filename.
        
        Format: a{action:02d}_s{subject:02d}_e{execution:02d}_skeleton.txt
        
        Returns:
            (action_id, subject_id, execution_id)
        """
        parts = filename.replace('_skeleton.txt', '').split('_')
        action_id = int(parts[0][1:])  # a01 -> 1
        subject_id = int(parts[1][1:])  # s01 -> 1
        execution_id = int(parts[2][1:])  # e01 -> 1
        return action_id, subject_id, execution_id
    
    def load_data(self, interpolate: bool = True, target_length: int = 50):
        """
        Load all skeleton files and split into train/test.
        
        Args:
            interpolate: Whether to interpolate sequences to fixed length
            target_length: Target length for interpolation
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        skeleton_files = sorted(self.data_dir.glob("a*_s*_e*_skeleton.txt"))
        logger.info(f"Found {len(skeleton_files)} skeleton files")
        
        train_raw = []
        test_raw = []
        
        for filepath in skeleton_files:
            action_id, subject_id, execution_id = self.extract_metadata(filepath.name)
            
            # Parse skeleton sequence
            sequence = self.parse_skeleton_file(filepath)
            if sequence is None or len(sequence) < 5:  # Skip very short sequences
                continue
            
            # Interpolate to fixed length if requested
            if interpolate and len(sequence) != target_length:
                sequence = self._interpolate_sequence(sequence, target_length)
            
            # Split by subject
            if subject_id in self.TRAIN_SUBJECTS:
                train_raw.append(sequence)
                self.train_labels.append(action_id - 1)  # 0-indexed labels
            elif subject_id in self.TEST_SUBJECTS:
                test_raw.append(sequence)
                self.test_labels.append(action_id - 1)
        
        logger.info(f"Loaded {len(train_raw)} train and {len(test_raw)} test sequences")
        
        # Compute normalization statistics from training data
        self._compute_normalization_stats(train_raw)
        
        # Normalize both train and test
        self.train_sequences = [self._normalize_sequence(seq) for seq in train_raw]
        self.test_sequences = [self._normalize_sequence(seq) for seq in test_raw]
        
        logger.info("Data loading complete")
        
    def _interpolate_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """
        Interpolate sequence to target length using linear interpolation.
        
        Args:
            sequence: Original sequence (T, 60)
            target_length: Target length
            
        Returns:
            Interpolated sequence (target_length, 60)
        """
        from scipy.interpolate import interp1d
        
        T = len(sequence)
        if T == target_length:
            return sequence
        
        # Original time indices
        t_old = np.linspace(0, 1, T)
        # New time indices
        t_new = np.linspace(0, 1, target_length)
        
        # Interpolate each dimension
        interpolator = interp1d(t_old, sequence, axis=0, kind='linear')
        interpolated = interpolator(t_new)
        
        return interpolated
    
    def _compute_normalization_stats(self, train_sequences: List[np.ndarray]):
        """
        Compute mean and std from training data for z-score normalization.
        
        Args:
            train_sequences: List of training sequences
        """
        # Stack all training frames
        all_frames = np.vstack(train_sequences)
        
        self.train_mean = np.mean(all_frames, axis=0)
        self.train_std = np.std(all_frames, axis=0) + 1e-8  # Avoid division by zero
        
        logger.info(f"Normalization stats computed from {len(all_frames)} frames")
    
    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization using training statistics.
        
        Args:
            sequence: Sequence to normalize
            
        Returns:
            Normalized sequence
        """
        return (sequence - self.train_mean) / self.train_std
    
    def save_split(self, output_path: str):
        """
        Save train/test split to JSON file for reproducibility.
        
        Args:
            output_path: Path to output JSON file
        """
        split_info = {
            "train_subjects": self.TRAIN_SUBJECTS,
            "test_subjects": self.TEST_SUBJECTS,
            "num_train": len(self.train_sequences),
            "num_test": len(self.test_sequences),
            "num_actions": 20,
            "frame_dim": self.FRAME_DIM,
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split info saved to {output_path}")
    
    def get_data(self) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
        """
        Get train and test data.
        
        Returns:
            (train_sequences, train_labels, test_sequences, test_labels)
        """
        return (
            self.train_sequences,
            self.train_labels,
            self.test_sequences,
            self.test_labels
        )


def load_msr_action3d(
    data_dir: str,
    interpolate: bool = True,
    target_length: int = 50,
    save_split: bool = True,
    split_path: str = "data/splits/msr_cs.json"
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Convenience function to load MSR Action3D dataset.
    
    Args:
        data_dir: Path to skeleton files
        interpolate: Whether to interpolate to fixed length
        target_length: Target sequence length
        save_split: Whether to save split info
        split_path: Path to save split info
        
    Returns:
        (train_sequences, train_labels, test_sequences, test_labels)
    """
    dataset = MSRAction3D(data_dir)
    dataset.load_data(interpolate=interpolate, target_length=target_length)
    
    if save_split:
        dataset.save_split(split_path)
    
    return dataset.get_data()


if __name__ == "__main__":
    # Test loading
    data_dir = "msr_action_data"
    
    train_seqs, train_labels, test_seqs, test_labels = load_msr_action3d(
        data_dir=data_dir,
        interpolate=True,
        target_length=50
    )
    
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_seqs)} sequences")
    print(f"Test: {len(test_seqs)} sequences")
    print(f"Frame dimension: {train_seqs[0].shape[1]}")
    print(f"Sample sequence shape: {train_seqs[0].shape}")
    print(f"Actions: {sorted(set(train_labels))}")
