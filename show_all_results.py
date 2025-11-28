"""
Comprehensive Model Comparison - All Models
"""
import json
import os

def main():
    # Load tree model results
    tree_results = {}
    tree_path = 'experiments/tree_models_comparison/results.json'
    if os.path.exists(tree_path):
        with open(tree_path) as f:
            tree_results = json.load(f)

    # Load neural network results (from summary.json)
    nn_results = {}
    nn_path = 'experiments/nn_enhanced_v2/summary.json'
    if os.path.exists(nn_path):
        with open(nn_path) as f:
            nn_results = json.load(f)

    # Combine all results
    all_models = []

    # Tree models
    for name, data in tree_results.items():
        all_models.append({
            'Model': name,
            'Type': 'Tree',
            'Params': '-',
            'Accuracy': data.get('accuracy', 0) * 100,
            'F1-Macro': data.get('f1_macro', 0) * 100,
            'Attack Recall': data.get('attack_recall', 0) * 100,
            'FAR': data.get('false_alarm', 0) * 100,
            'AUC': '-',
            'Train Time': data.get('time', 0),
        })

    # Neural network models (from summary.json structure)
    for name, data in nn_results.items():
        test_metrics = data.get('test_metrics', {})
        all_models.append({
            'Model': name,
            'Type': 'Neural Net',
            'Params': f"{data.get('n_parameters', 0):,}",
            'Accuracy': test_metrics.get('accuracy', 0),
            'F1-Macro': test_metrics.get('f1_macro', 0),
            'Attack Recall': test_metrics.get('detection_rate', 0),
            'FAR': test_metrics.get('false_alarm_rate', 0),
            'AUC': f"{test_metrics.get('auc', 0):.2f}%",
            'Train Time': data.get('training_time_seconds', 0),
        })

    # Print comprehensive table
    print('=' * 130)
    print('COMPREHENSIVE MODEL COMPARISON - ALL MODELS')
    print('=' * 130)
    header = f"{'Model':<15} {'Type':<12} {'Params':<12} {'Accuracy':>10} {'F1-Macro':>10} {'Attack Rec':>12} {'FAR':>8} {'AUC':>8} {'Time':>10}"
    print(header)
    print('-' * 130)

    # Sort by Attack Recall (most important for IDS)
    all_models.sort(key=lambda x: x['Attack Recall'], reverse=True)

    for m in all_models:
        time_str = f"{m['Train Time']:.1f}s" if m['Train Time'] else '-'
        auc_str = m['AUC'] if isinstance(m['AUC'], str) else '-'
        row = f"{m['Model']:<15} {m['Type']:<12} {str(m['Params']):<12} {m['Accuracy']:>9.2f}% {m['F1-Macro']:>9.2f}% {m['Attack Recall']:>11.2f}% {m['FAR']:>7.2f}% {auc_str:>10} {time_str:>10}"
        print(row)

    print('=' * 130)
    print()
    
    # Publication readiness check
    print('PUBLICATION READINESS CHECK')
    print('-' * 70)
    header2 = f"{'Model':<15} {'Acc>90%':>10} {'Recall>80%':>12} {'FAR<10%':>10} {'F1>85%':>10} {'Status':>12}"
    print(header2)
    print('-' * 70)

    for m in all_models:
        acc_pass = 'YES' if m['Accuracy'] >= 90 else 'NO'
        rec_pass = 'YES' if m['Attack Recall'] >= 80 else 'NO'
        far_pass = 'YES' if m['FAR'] <= 10 else 'NO'
        f1_pass = 'YES' if m['F1-Macro'] >= 85 else 'NO'
        
        passes = sum([m['Accuracy'] >= 90, m['Attack Recall'] >= 80, m['FAR'] <= 10, m['F1-Macro'] >= 85])
        
        if passes == 4:
            status = 'READY'
        elif passes == 3:
            status = 'CLOSE'
        else:
            status = 'NOT READY'
        
        row = f"{m['Model']:<15} {acc_pass:>10} {rec_pass:>12} {far_pass:>10} {f1_pass:>10} {status:>12}"
        print(row)

    print('-' * 70)
    print()
    
    # Summary insights
    print('=' * 70)
    print('KEY INSIGHTS FOR DECISION MAKING')
    print('=' * 70)
    
    # Find best models
    best_recall = max(all_models, key=lambda x: x['Attack Recall'])
    best_accuracy = max(all_models, key=lambda x: x['Accuracy'])
    best_f1 = max(all_models, key=lambda x: x['F1-Macro'])
    lowest_far = min(all_models, key=lambda x: x['FAR'])
    
    nn_models = [m for m in all_models if m['Type'] == 'Neural Net']
    best_nn = max(nn_models, key=lambda x: x['Attack Recall']) if nn_models else None
    
    print(f"\n1. BEST ATTACK DETECTION:")
    print(f"   {best_recall['Model']} ({best_recall['Type']}): {best_recall['Attack Recall']:.2f}% recall")
    
    print(f"\n2. BEST OVERALL ACCURACY:")
    print(f"   {best_accuracy['Model']} ({best_accuracy['Type']}): {best_accuracy['Accuracy']:.2f}%")
    
    print(f"\n3. LOWEST FALSE ALARM RATE:")
    print(f"   {lowest_far['Model']} ({lowest_far['Type']}): {lowest_far['FAR']:.2f}%")
    
    if best_nn:
        print(f"\n4. BEST NEURAL NETWORK (for edge deployment):")
        print(f"   {best_nn['Model']}: {best_nn['Attack Recall']:.2f}% recall, {best_nn['Params']} params")
    
    print()
    print('=' * 70)
    print('RECOMMENDATIONS')
    print('=' * 70)
    print("""
    FOR PUBLICATION:
    ----------------
    • Use LightGBM/XGBoost as primary models (96%+ attack recall)
    • Include LSTM as neural network baseline (77% recall, room to improve)
    • DS-CNN shows promise for edge deployment (59K params, 98.7% AUC)
    
    TO IMPROVE NEURAL NETWORKS:
    ---------------------------
    • Option 1: Train LSTM 50+ more epochs (was still improving)
    • Option 2: Lower classification threshold from 0.5 to 0.4
    • Option 3: Use knowledge distillation from LightGBM → DS-CNN
    
    FOR EDGE DEPLOYMENT (IoT/Embedded):
    -----------------------------------
    • DS-CNN: Only 59K params, fastest inference
    • Can be quantized to INT8 for 4x size reduction
    • Target: Distill LightGBM knowledge into DS-CNN
    """)

if __name__ == '__main__':
    main()
