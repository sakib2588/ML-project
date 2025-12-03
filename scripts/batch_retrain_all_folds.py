#!/usr/bin/env python3
"""
Batch retrain all folds (2-5) with SMOTE augmentation.
Automatically applies SMOTE + Focal Loss + Weighted Sampler to each fold.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

# Config
FOLDS = [2, 3, 4, 5]
REPO_ROOT = Path("/home/sakib/ids-compression")
PIPELINE_SCRIPT = REPO_ROOT / "scripts/complete_prephase2_pipeline.py"
DATA_BASE = REPO_ROOT / "data/processed"

def log(msg: str):
    """Print timestamped message"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def main():
    log("=" * 70)
    log("BATCH RETRAIN: FOLDS 2-5 WITH SMOTE")
    log("=" * 70)
    
    results_summary = {}
    
    for fold in FOLDS:
        log(f"\n{'='*70}")
        log(f"PROCESSING FOLD {fold}")
        log(f"{'='*70}")
        
        # Check if fold data exists
        fold_dir = DATA_BASE / f"cic_ids_2017_v{fold}"
        if not fold_dir.exists():
            log(f"❌ Fold {fold} data not found at {fold_dir}")
            continue
        
        # Set environment variable for fold
        env = {**dict(os.environ), "FOLD_NUM": str(fold)}
        
        # Run pipeline for this fold
        log(f"Starting SMOTE + retraining for Fold {fold}...")
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(PIPELINE_SCRIPT)],
                env=env,
                capture_output=False,
                timeout=14400,  # 4 hours max
                cwd=str(REPO_ROOT)
            )
            
            if result.returncode == 0:
                elapsed = (time.time() - start) / 3600
                log(f"✅ Fold {fold} completed successfully in {elapsed:.2f} hours")
                results_summary[f"Fold {fold}"] = "✅ PASS"
            else:
                log(f"❌ Fold {fold} failed with return code {result.returncode}")
                results_summary[f"Fold {fold}"] = "❌ FAIL"
        
        except subprocess.TimeoutExpired:
            log(f"❌ Fold {fold} timeout (exceeded 4 hours)")
            results_summary[f"Fold {fold}"] = "❌ TIMEOUT"
        except Exception as e:
            log(f"❌ Fold {fold} error: {e}")
            results_summary[f"Fold {fold}"] = f"❌ ERROR: {e}"
    
    # Summary
    log(f"\n{'='*70}")
    log("BATCH PROCESSING COMPLETE")
    log(f"{'='*70}")
    for fold_result, status in results_summary.items():
        log(f"{fold_result}: {status}")
    
    log(f"\nAll folds ready for Phase 2 compression!")
    log(f"Next: Implement cross-fold compression pipeline")

if __name__ == "__main__":
    import os
    main()
