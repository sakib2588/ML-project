"""
Complete Model Comparison for CIC-IDS2017

Compares Random Forest, XGBoost, and LightGBM with proper stratified sampling.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
import time
import json
from pathlib import Path

def run_comparison():
    print('='*70)
    print('COMPLETE MODEL COMPARISON - CIC-IDS2017')
    print('='*70)

    # Load and prepare data
    print('\nPreparing data with stratified split...')
    X_train = np.load('data/processed/cic_ids_2017/train/X.npy')
    y_train = np.load('data/processed/cic_ids_2017/train/y.npy')
    X_val = np.load('data/processed/cic_ids_2017/val/X.npy')
    y_val = np.load('data/processed/cic_ids_2017/val/y.npy')

    X_all = np.vstack([X_train, X_val]).reshape(-1, 15*8)
    y_all = np.concatenate([y_train, y_val])

    # Sample 150K for reasonable speed
    idx_0 = np.where(y_all == 0)[0]
    idx_1 = np.where(y_all == 1)[0]
    n_samples = 150000
    ratio = len(idx_0) / len(y_all)
    n_0, n_1 = int(n_samples * ratio), n_samples - int(n_samples * ratio)

    np.random.seed(42)
    idx_sample = np.concatenate([
        np.random.choice(idx_0, n_0, replace=False),
        np.random.choice(idx_1, n_1, replace=False)
    ])
    np.random.shuffle(idx_sample)

    X_sample, y_sample = X_all[idx_sample], y_all[idx_sample]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
    )

    print(f'Train: {len(X_tr)} | Test: {len(X_te)}')
    print(f'Train dist: {np.bincount(y_tr)} | Test dist: {np.bincount(y_te)}')
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    results = {}

    # ============ RANDOM FOREST ============
    print('\n' + '-'*70)
    print('1. RANDOM FOREST')
    print('-'*70)
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, 
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    rf_time = time.time() - start

    cm = confusion_matrix(y_te, y_pred)
    results['RandomForest'] = {
        'accuracy': float((y_pred == y_te).mean()),
        'f1_macro': float(f1_score(y_te, y_pred, average='macro')),
        'attack_recall': float(cm[1,1]/(cm[1,0]+cm[1,1])),
        'false_alarm': float(cm[0,1]/(cm[0,0]+cm[0,1])),
        'time': rf_time
    }
    r = results['RandomForest']
    print(f"Time: {rf_time:.1f}s | Acc: {r['accuracy']*100:.2f}% | F1: {r['f1_macro']*100:.2f}% | Attack Recall: {r['attack_recall']*100:.2f}%")

    # ============ XGBOOST ============
    print('\n' + '-'*70)
    print('2. XGBOOST')
    print('-'*70)
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=12, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, tree_method='hist',
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_tr, y_tr)
    y_pred = xgb_model.predict(X_te)
    xgb_time = time.time() - start

    cm = confusion_matrix(y_te, y_pred)
    results['XGBoost'] = {
        'accuracy': float((y_pred == y_te).mean()),
        'f1_macro': float(f1_score(y_te, y_pred, average='macro')),
        'attack_recall': float(cm[1,1]/(cm[1,0]+cm[1,1])),
        'false_alarm': float(cm[0,1]/(cm[0,0]+cm[0,1])),
        'time': xgb_time
    }
    r = results['XGBoost']
    print(f"Time: {xgb_time:.1f}s | Acc: {r['accuracy']*100:.2f}% | F1: {r['f1_macro']*100:.2f}% | Attack Recall: {r['attack_recall']*100:.2f}%")

    # ============ LIGHTGBM ============
    print('\n' + '-'*70)
    print('3. LIGHTGBM')
    print('-'*70)
    start = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=12, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    y_pred = lgb_model.predict(X_te)
    lgb_time = time.time() - start

    cm = confusion_matrix(y_te, y_pred)
    results['LightGBM'] = {
        'accuracy': float((y_pred == y_te).mean()),
        'f1_macro': float(f1_score(y_te, y_pred, average='macro')),
        'attack_recall': float(cm[1,1]/(cm[1,0]+cm[1,1])),
        'false_alarm': float(cm[0,1]/(cm[0,0]+cm[0,1])),
        'time': lgb_time
    }
    r = results['LightGBM']
    print(f"Time: {lgb_time:.1f}s | Acc: {r['accuracy']*100:.2f}% | F1: {r['f1_macro']*100:.2f}% | Attack Recall: {r['attack_recall']*100:.2f}%")

    # ============ SUMMARY ============
    print('\n' + '='*70)
    print('SUMMARY - MODEL COMPARISON')
    print('='*70)
    print(f"{'Model':<15} {'Accuracy':>10} {'F1-Macro':>10} {'AttackRecall':>12} {'FAR':>8} {'Time':>8}")
    print('-'*70)
    for name, m in results.items():
        print(f"{name:<15} {m['accuracy']*100:>9.2f}% {m['f1_macro']*100:>9.2f}% {m['attack_recall']*100:>11.2f}% {m['false_alarm']*100:>7.2f}% {m['time']:>7.1f}s")
    print('='*70)

    # Best model
    best_name = max(results.keys(), key=lambda x: results[x]['f1_macro'])
    best = results[best_name]
    print(f'\n🏆 BEST MODEL: {best_name}')
    print(f"   Detection Rate: {best['attack_recall']*100:.2f}%")
    print(f"   False Alarm Rate: {best['false_alarm']*100:.2f}%")
    print(f"   F1-Macro: {best['f1_macro']*100:.2f}%")

    # Save results
    output_dir = Path('experiments/tree_models_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Results saved to {output_dir / "results.json"}')

    return results


if __name__ == '__main__':
    run_comparison()
