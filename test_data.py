"""
Quick test script to verify MSR Action3D data structure and loading.
"""

import numpy as np
from pathlib import Path

def check_data_structure():
    """Check if data directory and files are correctly structured."""
    print("="*60)
    print("MSR Action3D Data Structure Check")
    print("="*60)
    
    data_dir = Path("msr_action_data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print(f"✅ Data directory found: {data_dir}")
    
    # Find skeleton files
    skeleton_files = list(data_dir.glob("a*_s*_e*_skeleton.txt"))
    print(f"✅ Found {len(skeleton_files)} skeleton files")
    
    if len(skeleton_files) == 0:
        print("❌ No skeleton files found!")
        return False
    
    # Parse filenames to get statistics
    actions = set()
    subjects = set()
    executions = set()
    
    for f in skeleton_files:
        parts = f.stem.replace('_skeleton', '').split('_')
        if len(parts) >= 3:
            actions.add(parts[0])
            subjects.add(parts[1])
            executions.add(parts[2])
    
    print(f"\nDataset Statistics:")
    print(f"  Actions: {len(actions)} - {sorted(actions)[:5]}...")
    print(f"  Subjects: {len(subjects)} - {sorted(subjects)}")
    print(f"  Executions: {len(executions)} - {sorted(executions)}")
    
    # Check a sample file
    sample_file = skeleton_files[0]
    print(f"\nSample file: {sample_file.name}")
    
    try:
        data = np.loadtxt(sample_file)
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        
        # Expected: (num_frames * 20, 4) where 4 = (x, y, z, confidence)
        if data.ndim == 2 and data.shape[1] == 4:
            num_joints = 20
            num_frames = data.shape[0] // num_joints
            print(f"  Estimated frames: {num_frames}")
            print(f"  Joints per frame: {num_joints}")
            print(f"✅ File format looks correct (20 joints × 4 values)")
        else:
            print(f"⚠️  Unexpected shape: {data.shape}")
        
        print(f"\nFirst few values:")
        print(data[:5])
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ Data structure check passed!")
    print("="*60)
    
    return True


def test_data_loading():
    """Test the actual data loading module."""
    print("\n" + "="*60)
    print("Testing Data Loading Module")
    print("="*60)
    
    try:
        from src.data.msr_action3d import MSRAction3D
        
        dataset = MSRAction3D("msr_action_data")
        dataset.load_data(interpolate=True, target_length=50)
        
        train_seqs, train_labels, test_seqs, test_labels = dataset.get_data()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"  Train sequences: {len(train_seqs)}")
        print(f"  Test sequences: {len(test_seqs)}")
        print(f"  Frame dimension: {train_seqs[0].shape[1]}")
        print(f"  Sample sequence shape: {train_seqs[0].shape}")
        print(f"  Unique actions: {len(set(train_labels))}")
        print(f"  Label range: {min(train_labels)} to {max(train_labels)}")
        
        # Check normalization
        all_frames = np.vstack(train_seqs)
        print(f"\n  Training data statistics:")
        print(f"    Mean: {np.mean(all_frames):.6f} (should be ~0)")
        print(f"    Std: {np.std(all_frames):.6f} (should be ~1)")
        
        print("\n" + "="*60)
        print("✅ Data loading module works correctly!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run checks
    structure_ok = check_data_structure()
    
    if structure_ok:
        print("\n" + "⏳ Running data loading test...")
        loading_ok = test_data_loading()
        
        if loading_ok:
            print("\n" + "="*60)
            print("🎉 ALL CHECKS PASSED!")
            print("="*60)
            print("\nYou're ready to run the pipeline:")
            print("  python main.py")
            print("\nOr test individual components:")
            print("  python src/dtw/core.py")
            print("  python src/quantum/swap_fidelity.py")
        else:
            print("\n⚠️  Data loading failed. Check dependencies.")
    else:
        print("\n⚠️  Data structure check failed.")
        print("\nPlease ensure skeleton files are in msr_action_data/")
