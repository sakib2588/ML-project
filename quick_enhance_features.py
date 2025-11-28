#!/usr/bin/env python
"""
Quick Feature Enhancement Script

This script creates an enhanced dataset with MORE features from CIC-IDS2017,
without re-running the full preprocessing pipeline.

Strategy: Load raw CSV, extract 40+ key features, create windows, save.
This is MUCH faster than re-running the full pipeline.

Usage: python quick_enhance_features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Key features to extract (40+ most informative for IDS)
FEATURE_COLUMNS = [
    # Duration and basic counts
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    
    # Packet length statistics
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    'Bwd Packet Length Max',
    ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    
    # Flow rates
    'Flow Bytes/s',
    ' Flow Packets/s',
    
    # Inter-arrival times
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Flow IAT Max',
    ' Flow IAT Min',
    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    
    # Packet rates
    'Fwd Packets/s',
    ' Bwd Packets/s',
    
    # Packet length aggregates
    ' Min Packet Length',
    ' Max Packet Length',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',
    
    # TCP Flags (very important for IDS!)
    'FIN Flag Count',
    ' SYN Flag Count',
    ' RST Flag Count',
    ' PSH Flag Count',
    ' ACK Flag Count',
    ' URG Flag Count',
    
    # Ratios and averages
    ' Down/Up Ratio',
    ' Average Packet Size',
    
    # Window sizes (important for attack detection)
    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',
    
    # Active/Idle
    'Active Mean',
    'Idle Mean',
    
    # Port (useful for port scans)
    ' Destination Port',
]


def load_and_process_csv(csv_path: Path, max_rows: int = None) -> pd.DataFrame:
    """Load a single CSV and extract features."""
    df = pd.read_csv(csv_path, nrows=max_rows)
    
    # Get available columns
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    # Extract features
    X = df[available_cols].copy()
    
    # Handle infinities and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Get labels
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    y = (df[label_col] != 'BENIGN').astype(int)
    
    return X, y


def create_windows(X: np.ndarray, y: np.ndarray, window_size: int = 15, stride: int = 5) -> tuple:
    """Create sliding windows."""
    n_samples = len(X)
    windows = []
    labels = []
    
    for i in range(0, n_samples - window_size + 1, stride):
        window = X[i:i + window_size]
        # Label: any malicious in window = malicious
        label = 1 if y[i:i + window_size].max() > 0 else 0
        windows.append(window)
        labels.append(label)
    
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    print("=" * 60)
    print("QUICK FEATURE ENHANCEMENT")
    print("=" * 60)
    
    raw_dir = Path("data/raw/cic_ids_2017")
    output_dir = Path("data/processed/cic_ids_2017_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Process each file
    all_X = []
    all_y = []
    
    # Limit rows per file for speed (100K per file = ~800K total)
    MAX_ROWS_PER_FILE = 100000
    
    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        try:
            X, y = load_and_process_csv(csv_file, max_rows=MAX_ROWS_PER_FILE)
            all_X.append(X.values)
            all_y.append(y.values)
            print(f"  {csv_file.name}: {len(X)} rows, {len(X.columns)} features")
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    # Combine
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"\nCombined data: {X_combined.shape}")
    print(f"Class distribution: {np.bincount(y_combined)}")
    print(f"Features: {X_combined.shape[1]}")
    
    # Scale features
    print("\nScaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Split chronologically (70/15/15)
    n = len(X_scaled)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    X_train_raw = X_scaled[:n_train]
    y_train_raw = y_combined[:n_train]
    X_val_raw = X_scaled[n_train:n_train + n_val]
    y_val_raw = y_combined[n_train:n_train + n_val]
    X_test_raw = X_scaled[n_train + n_val:]
    y_test_raw = y_combined[n_train + n_val:]
    
    # Create windows
    print("\nCreating windows...")
    X_train, y_train = create_windows(X_train_raw, y_train_raw, window_size=15, stride=3)
    X_val, y_val = create_windows(X_val_raw, y_val_raw, window_size=15, stride=3)
    X_test, y_test = create_windows(X_test_raw, y_test_raw, window_size=15, stride=3)
    
    print(f"\nFinal shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, dist={np.bincount(y_train)}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}, dist={np.bincount(y_val)}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, dist={np.bincount(y_test)}")
    
    # Save
    print("\nSaving...")
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "y.npy", y)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    print(f"\n✅ Enhanced dataset saved to {output_dir}")
    print(f"   Features: {X_train.shape[2]} (vs 8 in original)")
    print(f"   Window size: 15")
    print(f"   Ready for training!")
    
    return output_dir


if __name__ == "__main__":
    main()
