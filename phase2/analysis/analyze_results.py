#!/usr/bin/env python
"""
Phase 2 - Statistical Analysis and Reporting
=============================================

Aggregate results across stages and seeds, compute statistics,
generate publication-ready tables and plots.

Usage:
    # Generate full report
    python analyze_results.py --full
    
    # Generate specific stage analysis
    python analyze_results.py --stage stage1
    
    # Generate comparison plots
    python analyze_results.py --plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase2.config import SEEDS, ARTIFACTS_DIR, ACCURACY_TARGETS, HARDWARE_TARGETS
from phase2.utils import compute_ci, wilcoxon_test, cohens_d


# ============ LOAD RESULTS ============
def load_stage_results(stage: str, pattern: str = "results*.json") -> List[Dict]:
    """Load all results for a stage."""
    stage_dir = ARTIFACTS_DIR / stage
    results = []
    
    if not stage_dir.exists():
        return results
    
    for seed_dir in stage_dir.glob("seed*"):
        for result_file in seed_dir.glob(pattern):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    data['result_file'] = str(result_file)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    return results


def load_summary(stage: str, pattern: str = "summary*.json") -> Optional[Dict]:
    """Load stage summary if available."""
    stage_dir = ARTIFACTS_DIR / stage
    
    for summary_file in stage_dir.glob(pattern):
        try:
            with open(summary_file) as f:
                return json.load(f)
        except:
            pass
    
    return None


# ============ AGGREGATE METRICS ============
def aggregate_metrics(results: List[Dict], metrics: List[str]) -> Dict[str, Dict]:
    """Aggregate metrics across seeds."""
    aggregated = {}
    
    for metric in metrics:
        values = []
        for r in results:
            # Handle nested metrics
            if 'test_metrics' in r and metric in r['test_metrics']:
                values.append(r['test_metrics'][metric])
            elif metric in r:
                values.append(r[metric])
        
        if values:
            mean, std, ci = compute_ci(np.array(values))
            aggregated[metric] = {
                'mean': mean,
                'std': std,
                'ci_95': ci,
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values),
                'values': values
            }
    
    return aggregated


# ============ STATISTICAL TESTS ============
def compare_stages(
    stage_a: str,
    stage_b: str,
    metric: str = 'f1_macro'
) -> Dict:
    """Compare two stages using statistical tests."""
    
    results_a = load_stage_results(stage_a)
    results_b = load_stage_results(stage_b)
    
    # Extract values by seed
    values_a = {}
    values_b = {}
    
    for r in results_a:
        seed = r.get('seed')
        if seed and 'test_metrics' in r:
            values_a[seed] = r['test_metrics'].get(metric, 0)
    
    for r in results_b:
        seed = r.get('seed')
        if seed and 'test_metrics' in r:
            values_b[seed] = r['test_metrics'].get(metric, 0)
    
    # Pair by seed
    common_seeds = set(values_a.keys()) & set(values_b.keys())
    
    if len(common_seeds) < 3:
        return {'error': 'Not enough paired samples'}
    
    paired_a = [values_a[s] for s in sorted(common_seeds)]
    paired_b = [values_b[s] for s in sorted(common_seeds)]
    
    paired_a = np.array(paired_a)
    paired_b = np.array(paired_b)
    
    # Statistical tests
    stat, p_value = wilcoxon_test(paired_a, paired_b)
    effect = cohens_d(paired_a, paired_b)
    
    return {
        'stage_a': stage_a,
        'stage_b': stage_b,
        'metric': metric,
        'mean_a': np.mean(paired_a),
        'mean_b': np.mean(paired_b),
        'difference': np.mean(paired_b) - np.mean(paired_a),
        'wilcoxon_stat': stat,
        'p_value': p_value,
        'cohens_d': effect,
        'significant': p_value < 0.05,
        'n_pairs': len(common_seeds)
    }


# ============ GENERATE TABLES ============
def generate_main_results_table() -> pd.DataFrame:
    """Generate main results table across all stages."""
    
    stages = ['stage0', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5']
    stage_names = ['Baseline', 'KD', 'Pruned', 'KD-FT', 'QAT', 'TFLite']
    
    metrics = ['accuracy', 'f1_macro', 'detection_rate', 'false_alarm_rate', 
               'ddos_recall', 'portscan_recall']
    
    rows = []
    
    for stage, name in zip(stages, stage_names):
        results = load_stage_results(stage)
        
        if not results:
            continue
        
        agg = aggregate_metrics(results, metrics)
        
        row = {'Stage': name, 'N': len(results)}
        
        for metric in metrics:
            if metric in agg:
                mean = agg[metric]['mean']
                std = agg[metric]['std']
                row[metric] = f"{mean:.2f} ± {std:.2f}"
        
        # Add model size if available
        sizes = [r.get('model_size_mb', r.get('pruned', {}).get('size_mb', 0)) 
                 for r in results if r.get('model_size_mb') or r.get('pruned', {}).get('size_mb')]
        if sizes:
            row['Size (MB)'] = f"{np.mean(sizes):.3f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_per_class_table() -> pd.DataFrame:
    """Generate per-attack-class detection rates."""
    
    stages = ['stage0', 'stage1', 'stage3', 'stage4']
    stage_names = ['Baseline', 'KD', 'KD-FT', 'QAT']
    
    classes = ['DDoS', 'PortScan']
    
    rows = []
    
    for stage, name in zip(stages, stage_names):
        results = load_stage_results(stage)
        
        if not results:
            continue
        
        row = {'Stage': name}
        
        for cls in classes:
            metric = f'{cls.lower()}_recall'
            values = [r['test_metrics'].get(metric, 0) for r in results 
                     if 'test_metrics' in r and metric in r['test_metrics']]
            
            if values:
                row[cls] = f"{np.mean(values):.1f} ± {np.std(values):.1f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_compression_table() -> pd.DataFrame:
    """Generate compression statistics table."""
    
    rows = []
    
    # Stage 2 (pruning)
    prune_results = load_stage_results('stage2', 'pruning_*.json')
    for r in prune_results:
        if 'compression' in r:
            rows.append({
                'Method': f"Prune ({r.get('schedule_name', 'unknown')})",
                'Params Ratio': f"{r['compression']['param_ratio']*100:.1f}%",
                'Size Ratio': f"{r['compression']['size_ratio']*100:.1f}%",
                'F1 Drop': f"{r['compression']['f1_drop']:.2f}%"
            })
    
    # Stage 5 (TFLite)
    convert_results = load_stage_results('stage5', 'conversion_*.json')
    for r in convert_results:
        rows.append({
            'Method': f"TFLite ({r.get('quantize', 'unknown')})",
            'Params Ratio': 'N/A',
            'Size Ratio': f"{100/r.get('compression_ratio', 1):.1f}%",
            'F1 Drop': 'N/A'
        })
    
    return pd.DataFrame(rows)


# ============ GO/NO-GO ANALYSIS ============
def check_go_nogo() -> Dict:
    """Check all go/no-go criteria."""
    
    checks = []
    
    # Stage 0 -> Stage 1 (KD)
    stage0_results = load_stage_results('stage0')
    stage1_results = load_stage_results('stage1')
    
    if stage0_results and stage1_results:
        baseline_f1 = np.mean([r['test_metrics']['f1_macro'] for r in stage0_results 
                              if 'test_metrics' in r])
        kd_f1 = np.mean([r['test_metrics']['f1_macro'] for r in stage1_results 
                        if 'test_metrics' in r])
        
        improvement = kd_f1 - baseline_f1
        passed = improvement >= ACCURACY_TARGETS.kd_improvement_min
        
        checks.append({
            'transition': 'Stage 0 → Stage 1',
            'criterion': f'KD improves F1 by ≥{ACCURACY_TARGETS.kd_improvement_min}%',
            'value': f'{improvement:+.2f}%',
            'passed': passed
        })
    
    # Stage 1 -> Stage 2 (Prune)
    stage2_results = load_stage_results('stage2', 'pruning_*.json')
    
    for r in stage2_results:
        if 'compression' in r:
            drop = r['compression']['f1_drop']
            passed = drop <= ACCURACY_TARGETS.prune_immediate_drop_max
            
            checks.append({
                'transition': f"Stage 1 → Stage 2 ({r.get('schedule_name', '')})",
                'criterion': f'Immediate F1 drop ≤{ACCURACY_TARGETS.prune_immediate_drop_max}%',
                'value': f'{drop:.2f}%',
                'passed': passed
            })
    
    # Stage 3 -> Stage 4 (QAT)
    stage3_results = load_stage_results('stage3')
    stage4_results = load_stage_results('stage4')
    
    if stage3_results and stage4_results:
        stage3_f1 = np.mean([r['test_metrics']['f1_macro'] for r in stage3_results 
                           if 'test_metrics' in r])
        stage4_f1 = np.mean([r['test_metrics']['f1_macro'] for r in stage4_results 
                           if 'test_metrics' in r])
        
        drop = stage3_f1 - stage4_f1
        passed = abs(drop) <= ACCURACY_TARGETS.qat_drop_max
        
        checks.append({
            'transition': 'Stage 3 → Stage 4',
            'criterion': f'QAT F1 drop ≤{ACCURACY_TARGETS.qat_drop_max}%',
            'value': f'{drop:.2f}%',
            'passed': passed
        })
    
    # Critical recall checks
    for stage in ['stage3', 'stage4']:
        results = load_stage_results(stage)
        
        if results:
            ddos_recalls = [r['test_metrics'].get('ddos_recall', 0) for r in results 
                          if 'test_metrics' in r]
            portscan_recalls = [r['test_metrics'].get('portscan_recall', 0) for r in results 
                               if 'test_metrics' in r]
            
            if ddos_recalls:
                passed = np.mean(ddos_recalls) >= ACCURACY_TARGETS.critical_recall_min
                checks.append({
                    'transition': f'{stage} Critical',
                    'criterion': f'DDoS recall >{ACCURACY_TARGETS.critical_recall_min}%',
                    'value': f'{np.mean(ddos_recalls):.1f}%',
                    'passed': passed
                })
            
            if portscan_recalls:
                passed = np.mean(portscan_recalls) >= ACCURACY_TARGETS.critical_recall_min
                checks.append({
                    'transition': f'{stage} Critical',
                    'criterion': f'PortScan recall >{ACCURACY_TARGETS.critical_recall_min}%',
                    'value': f'{np.mean(portscan_recalls):.1f}%',
                    'passed': passed
                })
    
    return {
        'checks': checks,
        'all_passed': all(c['passed'] for c in checks),
        'n_passed': sum(1 for c in checks if c['passed']),
        'n_total': len(checks)
    }


# ============ GENERATE PLOTS ============
def generate_plots(output_dir: Path):
    """Generate all publication plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping plots.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. F1 vs Compression plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['stage0', 'stage1', 'stage2', 'stage3', 'stage4']
    stage_names = ['Baseline', 'KD', 'Pruned', 'KD-FT', 'QAT']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for stage, name, color in zip(stages, stage_names, colors):
        results = load_stage_results(stage)
        
        if not results:
            continue
        
        f1_values = [r['test_metrics']['f1_macro'] for r in results if 'test_metrics' in r]
        
        # Get compression (relative to baseline)
        if stage == 'stage0':
            compression = [1.0] * len(f1_values)
        else:
            # Estimate based on params
            baseline_params = 85000  # Approximate
            params = [r.get('n_params', r.get('pruned', {}).get('params', baseline_params)) 
                     for r in results]
            compression = [baseline_params / max(p, 1) for p in params]
        
        ax.scatter(compression, f1_values, label=name, c=color, s=100, alpha=0.7)
    
    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('F1-Macro (%)', fontsize=12)
    ax.set_title('F1 vs Compression Ratio', fontsize=14)
    ax.legend()
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Target')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_vs_compression.png', dpi=300)
    plt.close()
    
    # 2. Per-stage F1 boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = []
    labels = []
    
    for stage, name in zip(stages, stage_names):
        results = load_stage_results(stage)
        f1_values = [r['test_metrics']['f1_macro'] for r in results if 'test_metrics' in r]
        
        if f1_values:
            data.append(f1_values)
            labels.append(name)
    
    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax.set_ylabel('F1-Macro (%)', fontsize=12)
        ax.set_title('F1-Macro Distribution by Stage', fontsize=14)
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'f1_by_stage.png', dpi=300)
        plt.close()
    
    # 3. Critical recall bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(stages))
    width = 0.35
    
    ddos_means = []
    portscan_means = []
    
    for stage in stages:
        results = load_stage_results(stage)
        
        ddos = [r['test_metrics'].get('ddos_recall', 0) for r in results if 'test_metrics' in r]
        portscan = [r['test_metrics'].get('portscan_recall', 0) for r in results if 'test_metrics' in r]
        
        ddos_means.append(np.mean(ddos) if ddos else 0)
        portscan_means.append(np.mean(portscan) if portscan else 0)
    
    ax.bar(x - width/2, ddos_means, width, label='DDoS', color='steelblue')
    ax.bar(x + width/2, portscan_means, width, label='PortScan', color='darkorange')
    
    ax.axhline(y=98, color='red', linestyle='--', alpha=0.7, label='Target (98%)')
    
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Critical Class Recall by Stage', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.legend()
    ax.set_ylim(80, 102)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_recall.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


# ============ PER-CLASS DEGRADATION TRACKING (NEW - Per feedback) ============
def generate_per_class_degradation_plot(output_dir: Path):
    """
    Plot how each attack class degrades through compression stages.
    
    CRITICAL: This visualization helps identify if specific attack types
    (e.g., Bot, SSH-Patator) degrade disproportionately during compression.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping per-class plots.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define attack types to track
    attack_types = [
        'DDoS', 'PortScan', 'Bot', 'SSH-Patator', 'FTP-Patator',
        'DoS Hulk', 'DoS Slowloris', 'DoS GoldenEye', 'DoS slowhttptest',
        'Infiltration', 'Web Attack'
    ]
    
    stages = ['stage0', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5']
    stage_names = ['Baseline', 'KD', 'Pruned', 'KD-FT', 'QAT', 'TFLite']
    
    # Collect per-class detection rates for each stage
    per_class_by_stage = {stage: {} for stage in stages}
    
    for stage in stages:
        results = load_stage_results(stage)
        
        for r in results:
            # Check for per-attack metrics in results
            per_attack = r.get('test_metrics', {}).get('per_attack_detection_rate', {})
            
            if not per_attack:
                # Try alternative locations
                per_attack = r.get('per_attack_detection_rate', {})
            
            for attack, dr in per_attack.items():
                if attack not in per_class_by_stage[stage]:
                    per_class_by_stage[stage][attack] = []
                per_class_by_stage[stage][attack].append(dr)
    
    # Calculate means for each stage
    per_class_means = {attack: [] for attack in attack_types}
    per_class_stds = {attack: [] for attack in attack_types}
    
    for stage in stages:
        for attack in attack_types:
            if attack in per_class_by_stage[stage] and per_class_by_stage[stage][attack]:
                per_class_means[attack].append(np.mean(per_class_by_stage[stage][attack]))
                per_class_stds[attack].append(np.std(per_class_by_stage[stage][attack]))
            else:
                per_class_means[attack].append(np.nan)
                per_class_stds[attack].append(np.nan)
    
    # Plot 1: Line plot of detection rate across stages
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / len(attack_types)) for i in range(len(attack_types))]
    critical_attacks = ['DDoS', 'PortScan', 'Bot', 'SSH-Patator']
    
    for i, attack in enumerate(attack_types):
        if any(not np.isnan(v) for v in per_class_means[attack]):
            linestyle = '-' if attack in critical_attacks else '--'
            linewidth = 2.5 if attack in critical_attacks else 1.5
            marker = 'o' if attack in critical_attacks else 's'
            
            ax.plot(
                stage_names, per_class_means[attack],
                label=attack, color=colors[i],
                linestyle=linestyle, linewidth=linewidth, marker=marker
            )
    
    ax.axhline(y=98, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Critical Target (98%)')
    ax.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Min Target (90%)')
    
    ax.set_xlabel('Compression Stage', fontsize=12)
    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_title('Per-Attack Detection Rate Across Compression Stages', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap of detection rates
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create matrix for heatmap
    matrix = np.array([per_class_means[attack] for attack in attack_types if any(not np.isnan(v) for v in per_class_means[attack])])
    attack_labels = [attack for attack in attack_types if any(not np.isnan(v) for v in per_class_means[attack])]
    
    if len(matrix) > 0:
        sns.heatmap(
            matrix,
            annot=True, fmt='.1f',
            xticklabels=stage_names,
            yticklabels=attack_labels,
            cmap='RdYlGn',
            vmin=0, vmax=100,
            ax=ax
        )
        
        ax.set_title('Detection Rate Heatmap Across Stages', fontsize=14)
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Attack Type', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_heatmap.png', dpi=300)
        plt.close()
    
    # Plot 3: Degradation from baseline (delta plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline_idx = 0  # stage0
    
    for i, attack in enumerate(critical_attacks):
        if len(per_class_means[attack]) > baseline_idx and not np.isnan(per_class_means[attack][baseline_idx]):
            baseline = per_class_means[attack][baseline_idx]
            deltas = [v - baseline if not np.isnan(v) else np.nan for v in per_class_means[attack]]
            
            ax.plot(stage_names, deltas, label=attack, marker='o', linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=-5, color='orange', linestyle='--', alpha=0.7, label='Warning (-5%)')
    ax.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='Critical (-10%)')
    
    ax.set_xlabel('Compression Stage', fontsize=12)
    ax.set_ylabel('Detection Rate Change from Baseline (%)', fontsize=12)
    ax.set_title('Critical Attack Detection Rate Degradation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_class_degradation.png', dpi=300)
    plt.close()
    
    print(f"Per-class degradation plots saved to: {output_dir}")
    
    # Generate summary table
    summary_data = []
    for attack in attack_types:
        if any(not np.isnan(v) for v in per_class_means[attack]):
            valid_values = [v for v in per_class_means[attack] if not np.isnan(v)]
            if len(valid_values) >= 2:
                baseline = per_class_means[attack][baseline_idx] if not np.isnan(per_class_means[attack][baseline_idx]) else valid_values[0]
                final = valid_values[-1]
                
                summary_data.append({
                    'Attack': attack,
                    'Baseline DR': f"{baseline:.1f}%",
                    'Final DR': f"{final:.1f}%",
                    'Drop': f"{baseline - final:.1f}%",
                    'Status': '✅ OK' if (baseline - final) < 5 else '⚠️ WARNING' if (baseline - final) < 10 else '❌ CRITICAL'
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'per_class_summary.csv', index=False)
        print(f"Per-class summary saved to: {output_dir / 'per_class_summary.csv'}")
        
        return summary_df
    
    return None


# ============ GENERATE REPORT ============
def generate_full_report(output_dir: Path):
    """Generate complete analysis report."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Phase 2 Compression Pipeline - Analysis Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        ""
    ]
    
    # Main results table
    report_lines.append("### Main Results")
    report_lines.append("")
    
    main_table = generate_main_results_table()
    if not main_table.empty:
        report_lines.append(main_table.to_markdown(index=False))
    report_lines.append("")
    
    # Per-class results
    report_lines.append("### Per-Class Detection Rates")
    report_lines.append("")
    
    per_class_table = generate_per_class_table()
    if not per_class_table.empty:
        report_lines.append(per_class_table.to_markdown(index=False))
    report_lines.append("")
    
    # Compression results
    report_lines.append("### Compression Results")
    report_lines.append("")
    
    compression_table = generate_compression_table()
    if not compression_table.empty:
        report_lines.append(compression_table.to_markdown(index=False))
    report_lines.append("")
    
    # Go/No-Go checks
    report_lines.append("### Go/No-Go Checks")
    report_lines.append("")
    
    go_nogo = check_go_nogo()
    
    for check in go_nogo['checks']:
        status = "✓" if check['passed'] else "✗"
        report_lines.append(f"- {status} **{check['transition']}**: {check['criterion']} → {check['value']}")
    
    report_lines.append("")
    report_lines.append(f"**Overall: {go_nogo['n_passed']}/{go_nogo['n_total']} checks passed**")
    
    if go_nogo['all_passed']:
        report_lines.append("")
        report_lines.append("🎉 **All checks passed - Ready for deployment!**")
    
    # Statistical comparisons
    report_lines.append("")
    report_lines.append("### Statistical Comparisons")
    report_lines.append("")
    
    comparisons = [
        ('stage0', 'stage1'),  # Baseline vs KD
        ('stage1', 'stage3'),  # KD vs KD-FT
        ('stage3', 'stage4'),  # KD-FT vs QAT
    ]
    
    for stage_a, stage_b in comparisons:
        comparison = compare_stages(stage_a, stage_b, 'f1_macro')
        
        if 'error' not in comparison:
            sig = "significant" if comparison['significant'] else "not significant"
            report_lines.append(
                f"- **{stage_a} vs {stage_b}**: "
                f"Δ = {comparison['difference']:+.2f}%, "
                f"p = {comparison['p_value']:.4f} ({sig}), "
                f"d = {comparison['cohens_d']:.3f}"
            )
    
    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")
    
    # Save tables as CSV
    main_table.to_csv(output_dir / "main_results.csv", index=False)
    per_class_table.to_csv(output_dir / "per_class_results.csv", index=False)
    compression_table.to_csv(output_dir / "compression_results.csv", index=False)
    
    # Save go/no-go as JSON
    with open(output_dir / "go_nogo.json", 'w') as f:
        json.dump(go_nogo, f, indent=2)
    
    # Generate plots
    generate_plots(output_dir / "plots")
    
    # Generate per-class degradation analysis (NEW)
    print("\nGenerating per-class degradation analysis...")
    per_class_summary = generate_per_class_degradation_plot(output_dir / "plots")
    
    if per_class_summary is not None:
        report_lines.append("")
        report_lines.append("### Per-Class Degradation Summary")
        report_lines.append("")
        report_lines.append(per_class_summary.to_markdown(index=False))
    
    return report_path


# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Phase 2 Analysis')
    parser.add_argument('--full', action='store_true', help='Generate full report')
    parser.add_argument('--stage', type=str, help='Analyze specific stage')
    parser.add_argument('--plots', action='store_true', help='Generate plots only')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = ARTIFACTS_DIR / args.output
    
    print(f"\n{'='*70}")
    print("PHASE 2 - ANALYSIS AND REPORTING")
    print(f"{'='*70}")
    
    if args.full:
        generate_full_report(output_dir)
    
    elif args.plots:
        generate_plots(output_dir / "plots")
    
    elif args.stage:
        results = load_stage_results(args.stage)
        print(f"\nLoaded {len(results)} results for {args.stage}")
        
        if results:
            metrics = ['accuracy', 'f1_macro', 'detection_rate', 'false_alarm_rate']
            agg = aggregate_metrics(results, metrics)
            
            print(f"\nAggregated Metrics:")
            for metric, stats in agg.items():
                print(f"  {metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    else:
        # Quick summary
        print("\nQuick Summary:")
        for stage in ['stage0', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5']:
            results = load_stage_results(stage)
            if results:
                f1_values = [r['test_metrics']['f1_macro'] for r in results 
                            if 'test_metrics' in r]
                if f1_values:
                    print(f"  {stage}: {np.mean(f1_values):.2f}% ± {np.std(f1_values):.2f}% (n={len(f1_values)})")


if __name__ == '__main__':
    main()
