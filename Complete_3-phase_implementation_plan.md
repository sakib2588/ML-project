# Complete 3-Phase Research Implementation Plan
## Compressing DS_1D_CNN-based Network Intrusion Detection for Raspberry Pi 4

**Project Duration**: 20 weeks (5 months) - Extended from 12-14 weeks for thoroughness  
**Revised Focus**: Systematic compression-attack degradation analysis with multi-dataset fusion

---

## 🎯 RESEARCH POSITIONING & NOVELTY

### Core Contribution (What Makes This Publishable)

**NOT**: "We compressed a model for Pi4" ← This exists (2024-2025)  
**YES**: "We systematically analyzed how compression affects different attack families across merged datasets and quantified cross-dataset generalization under resource constraints"

### Unique Value Propositions

1. **Per-Attack-Family Compression Impact Study**: First systematic analysis of which attacks survive compression and which fail (by stage: KD → Pruning → QAT)
2. **Cross-Dataset Generalization Under Compression**: Quantified how dataset fusion affects model robustness when compressed
3. **Compression-Bias Interaction Analysis**: How dataset artifacts amplify under compression
4. **Reproducible Benchmark Suite**: Complete pipeline with provenance tracking for IDS compression research

---

## 📊 PHASE 1: FOUNDATION & BASELINE (WEEKS 1-6)

**Goal**: Build clean, reproducible data pipeline and baseline models with comprehensive documentation

### Week 1-2: Data Acquisition & Quality Assessment

#### Deliverables
- Downloaded and validated datasets with checksums
- Comprehensive data quality report
- Initial per-class distribution analysis
- Dataset decision document (keep/merge/drop classes)

#### Tasks

**1.1 Dataset Download & Validation** (Days 1-2)
```
Priority A: NF-UQ-NIDS-v2
- Source: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- Size: ~76M records, 43 features
- Validate: Record count, column names, attack distribution

Priority A: NF-CSE-CIC-IDS2018-v2  
- Source: Same repository
- Size: ~19M records, 43 features
- Validate: Feature alignment with NF-UQ

Priority B (Optional): NF-ToN-IoT-v2 or NF-BoT-IoT-v2
- Use ONLY if needed for specific attack types
- Must have NetFlow v2 (43 features)
```

**1.2 Data Inventory Script** (Days 3-4)
```
Script: scripts/01_data_inventory.py

Outputs:
- data_inventory_report.json
  {
    "dataset": "NF-UQ-NIDS-v2",
    "total_records": 75987976,
    "features": 43,
    "attack_categories": {
      "DDoS": {"count": 12000000, "percentage": 15.8},
      "DoS": {"count": 8000000, "percentage": 10.5},
      ...
    },
    "missing_values": {...},
    "data_types": {...}
  }

- per_class_histogram.png
- feature_distributions.png
- missing_value_heatmap.png
```

**1.3 Class Decision Matrix** (Days 5-7)
```
Create: reports/phase1/class_decision_matrix.csv

For each attack class:
- Total samples across all datasets
- Samples per dataset (provenance)
- Decision: KEEP (>2000) / MERGE (500-2000) / DROP (<500)
- Justification
- Target post-augmentation count

Example:
Class          | NF-UQ | NF-CSE | Total  | Decision | Justification
---------------|-------|--------|--------|----------|------------------
DDoS           | 10M   | 2M     | 12M    | KEEP     | High impact, abundant
SSH-Patator    | 5000  | 3000   | 8000   | KEEP     | Above threshold
FTP-Patator    | 300   | 200    | 500    | MERGE    | Combine as Brute-Force
MITM           | 50    | 30     | 80     | DROP     | Insufficient samples
```

**DECISION GATE 1**: Review class decisions. Classes with <500 samples will be dropped or merged. Document all decisions.

---

### Week 3-4: Feature Engineering & Canonical Schema

#### Deliverables
- Unified canonical feature schema (43 features)
- Cleaned, merged dataset with provenance
- Feature correlation and importance analysis
- Preprocessing pipeline code

#### Tasks

**2.1 Canonical Schema Implementation** (Days 8-10)
```
Script: scripts/02_canonical_schema.py

Input: Raw datasets (CSVs)
Output: data/canonical/merged_canonical.parquet

Canonical Features (43 NetFlow features):
1. Flow duration
2. Total packets (fwd)
3. Total packets (bwd)
4. Total bytes (fwd)
5. Total bytes (bwd)
... (complete 43-feature list)

Critical additions:
- dataset_origin: {'nf-uq', 'nf-cse', ...}
- original_label: Original attack label
- canonical_label: Mapped attack label
- sample_id: Unique identifier
- is_benign: Boolean flag
```

**2.2 Feature Alignment Validation** (Days 11-12)
```
Script: scripts/03_feature_validation.py

Checks:
1. All datasets have same 43 features (strict mode)
2. Feature distributions by dataset (detect drift)
3. Missing value patterns by dataset
4. Categorical encoding consistency

Output: 
- feature_alignment_report.html
- distribution_comparison_plots/
  - feature_01_by_dataset.png
  - feature_02_by_dataset.png
  - ...
```

**2.3 Data Cleaning Pipeline** (Days 13-14)
```
Script: scripts/04_data_cleaning.py

Steps:
1. Remove invalid/incomplete flows
2. Handle missing values (document strategy per feature)
3. Remove exact duplicates (keep provenance for near-duplicates)
4. Cap outliers (document thresholds)
5. Normalize numeric features (preserve means/stds for later)
6. Encode categorical features (save encoders)

Output:
- data/cleaned/merged_cleaned.parquet
- cleaning_report.json (documents all transformations)
- feature_scalers.pkl (for inference)
- categorical_encoders.pkl
```

---

### Week 5-6: Baseline Models & Initial Evaluation

#### Deliverables
- Three trained baseline models (MLP, LSTM, DS-CNN)
- Initial performance benchmarks
- FLOPs and parameter counts
- Phase 1 decision report

#### Tasks

**3.1 Dataset Splitting** (Days 15-16)
```
Script: scripts/05_dataset_split.py

Strategy:
1. Stratified split by canonical_label
2. Hold out ENTIRE dataset for cross-dataset generalization
   - Train on: NF-UQ-NIDS-v2
   - Test on: NF-CSE-CIC-IDS2018-v2
   
3. Within training dataset:
   - Train: 70%
   - Validation: 15%
   - Test (same-dataset): 15%

4. Maintain class balance in each split

Output:
- data/splits/train.parquet
- data/splits/val.parquet  
- data/splits/test_same.parquet
- data/splits/test_cross.parquet (different dataset)
- split_statistics.json
```

**3.2 Baseline Model Implementation** (Days 17-20)

**Model A: Small MLP**
```python
# models/mlp.py

class SmallMLP(BaseModel):
    """
    Architecture:
    - Input: Flattened flow vector (43 features × window_size)
    - Dense(128, ReLU) + Dropout(0.3)
    - Dense(64, ReLU) + Dropout(0.3)
    - Dense(num_classes, Softmax)
    
    Target: ~50K parameters
    """
```

**Model B: Small LSTM**
```python
# models/lstm.py

class SmallLSTM(BaseModel):
    """
    Architecture:
    - Input: Sequence of flows (43 features × window_size)
    - LSTM(64, return_sequences=True)
    - LSTM(32, return_sequences=False)  
    - Dense(num_classes, Softmax)
    
    Target: ~100K parameters
    """
```

**Model C: DS-1D-CNN (Primary Candidate)**
```python
# models/ds_cnn.py

class DS_CNN_Small(BaseModel):
    """
    Architecture:
    - Input: 1D flow sequence (43 features × window_size)
    - DepthwiseConv1D(64, kernel=3) + BatchNorm + ReLU
    - PointwiseConv1D(64)
    - MaxPool1D(2)
    - DepthwiseConv1D(128, kernel=3) + BatchNorm + ReLU
    - PointwiseConv1D(128)
    - GlobalAveragePooling1D
    - Dense(num_classes, Softmax)
    
    Target: ~80K parameters
    """
```

**3.3 Training Configuration** (Days 21-24)
```yaml
# configs/baseline_training.yaml

common:
  window_size: 15  # consecutive flows
  batch_size: 256
  epochs: 50
  early_stopping_patience: 10
  learning_rate: 0.001
  
loss_config:
  binary_loss: binary_crossentropy
  multiclass_loss: categorical_crossentropy
  class_weights: auto  # Computed from class distribution
  
optimizer:
  type: Adam
  beta_1: 0.9
  beta_2: 0.999
  
callbacks:
  - ModelCheckpoint (save best by val_macro_f1)
  - EarlyStopping
  - ReduceLROnPlateau
  - CSVLogger
```

**3.4 Comprehensive Evaluation** (Days 25-28)
```
Script: scripts/06_evaluate_baselines.py

Metrics (per model):
1. Overall: Accuracy, Precision, Recall, F1
2. Binary: Accuracy, Precision, Recall, F1, ROC-AUC
3. Multi-class: Macro-F1, Per-class Precision/Recall/F1
4. Confusion matrices (same-dataset and cross-dataset)
5. ROC curves (binary and one-vs-rest)

Model Characteristics:
1. Total parameters
2. FLOPs (using thop or similar)
3. Model size on disk (SavedModel and TFLite)
4. Inference time (CPU, batch_size=1, 100 warmup + 1000 runs)

Output:
- experiments/phase1_baselines/
  - mlp/
    - model.h5
    - training_history.csv
    - evaluation_metrics.json
    - confusion_matrix.png
    - roc_curves.png
  - lstm/
    - ...
  - ds_cnn/
    - ...
  - comparison_plots/
    - accuracy_comparison.png
    - size_vs_accuracy.png
    - flops_vs_accuracy.png
    - inference_time_comparison.png
```

**3.5 Phase 1 Decision Report** (Days 29-30)
```
Document: reports/phase1_decision_report.md

Sections:
1. Dataset Quality Assessment
   - Total samples kept/merged/dropped
   - Class distribution analysis
   - Data quality issues found and resolved

2. Baseline Model Performance
   - Performance table (all models × all metrics)
   - Same-dataset vs cross-dataset performance gaps
   - Model size and speed comparison

3. Architecture Selection for Phase 2
   - Selected: DS-1D-CNN (or other with justification)
   - Reasoning: Balance of accuracy, size, speed
   - Expected compression potential

4. Risk Assessment
   - Classes at risk under compression
   - Cross-dataset generalization concerns
   - Timeline risks

5. Go/No-Go Decision
   - Proceed to Phase 2: YES/NO
   - Conditions for proceeding
```

**DECISION GATE 2**: Review Phase 1 report. Baseline DS-CNN should achieve:
- Binary accuracy ≥97%
- Macro-F1 ≥90%
- Inference time ≤50ms (CPU, unoptimized)
- Model size <500KB uncompressed

If not met, revisit architecture or data strategy.

---

## 🔬 PHASE 2: COMPRESSION PIPELINE (WEEKS 7-14)

**Goal**: Implement and analyze multi-stage compression with per-attack-family impact tracking

### Week 7-8: Teacher Model & Knowledge Distillation Setup

#### Deliverables
- High-capacity teacher model
- KD training pipeline
- Initial compressed student models
- Per-attack KD impact analysis

#### Tasks

**4.1 Teacher Model Training** (Days 31-35)
```python
# models/teacher.py

class TeacherModel(BaseModel):
    """
    High-capacity model for knowledge distillation
    
    Architecture (example for DS-CNN-based teacher):
    - 4-5 DS-Conv blocks instead of 2-3
    - Wider channels (256, 512)
    - Additional dense layers
    - Multi-task heads (binary + multiclass)
    
    Target: 500K-1M parameters
    Acceptable training time: 2-3 days on GPU
    """

# Training:
# - Use full training set
# - Aggressive data augmentation
# - Ensemble of 3 teachers (optional, for robust KD)
# - Save soft labels (logits) for training set
```

**4.2 Knowledge Distillation Implementation** (Days 36-40)
```python
# training/kd_trainer.py

class KDTrainer:
    """
    Knowledge Distillation training loop
    
    Loss = alpha * KD_loss + (1-alpha) * CE_loss
    
    KD_loss = KL_divergence(
        student_logits / T,
        teacher_logits / T
    ) * (T^2)
    
    Parameters:
    - Temperature T: [2.0, 4.0, 6.0] (grid search)
    - Alpha: [0.5, 0.7, 0.9] (grid search)
    - Student size: [30K, 50K, 80K] params
    """

# Experiment matrix:
# - 3 temperatures × 3 alphas × 3 sizes = 27 experiments
# - Quick training (20 epochs) to find best config
# - Full training (50 epochs) for best 3 configs
```

**4.3 Per-Attack KD Impact Analysis** (Days 41-42)
```
Script: scripts/07_kd_impact_analysis.py

For each attack category:
1. Teacher performance (precision, recall, F1)
2. Student performance (same metrics)
3. Knowledge transfer efficiency = (student_f1 / teacher_f1)
4. Hard sample analysis (which samples did student fail?)

Output:
- reports/phase2/kd_impact_by_attack.csv
- plots/kd_transfer_efficiency_by_attack.png
- analysis_document.md (identifies vulnerable attack types)

Key Question: Which attacks suffer most from KD compression?
```

---

### Week 9-11: Structured Pruning with Attack-Family Tracking

#### Deliverables
- Iteratively pruned models (10%, 20%, 30%, 40%, 50%)
- Per-attack performance degradation curves
- Pruning strategy analysis

#### Tasks

**5.1 Importance-Based Filter Pruning** (Days 43-50)
```python
# compression/structured_pruning.py

class StructuredPruner:
    """
    Iterative structured pruning with importance scoring
    
    Importance score = L1_norm(weights) × Gradient_sensitivity
    
    Strategy:
    - Prune 10% of filters with lowest importance
    - Fine-tune for 5 epochs with KD (teacher still providing guidance)
    - Evaluate on validation set
    - Repeat until target compression or quality degradation
    
    Stopping criteria:
    - Macro-F1 drops >5% from unpruned model
    - Any major attack class drops >10% in recall
    """

# Pruning schedule:
pruning_configs = [
    {"ratio": 0.1, "finetune_epochs": 5, "name": "prune_10"},
    {"ratio": 0.2, "finetune_epochs": 5, "name": "prune_20"},
    {"ratio": 0.3, "finetune_epochs": 7, "name": "prune_30"},
    {"ratio": 0.4, "finetune_epochs": 7, "name": "prune_40"},
    {"ratio": 0.5, "finetune_epochs": 10, "name": "prune_50"},
]
```

**5.2 Per-Stage Attack Impact Tracking** (Days 51-55)
```
Script: scripts/08_pruning_attack_tracking.py

For each pruning stage:
1. Evaluate ALL attack categories
2. Track metrics:
   - Per-class precision, recall, F1
   - Confusion matrix evolution
   - Hard sample difficulty distribution
   
Output:
- results/pruning_stages/
  - prune_10_metrics.json
  - prune_20_metrics.json
  - ...
  - attack_degradation_curves.png
  - vulnerable_attacks_heatmap.png

Key Analysis:
- Which attacks survive 50% pruning?
- Which attacks fail first?
- Is degradation gradual or sudden?
```

**5.3 Pruning Strategy Comparison** (Days 56-58)
```
Experiment: Compare pruning strategies

Strategies:
A. Magnitude-based (L1 norm only)
B. Importance-based (L1 × gradient)
C. Channel-wise (entire channels)
D. Layer-wise (distribute pruning unevenly)

Metrics:
- Final model size
- Accuracy retention
- Per-attack degradation patterns

Output:
- reports/phase2/pruning_strategy_comparison.md
- Recommendation: Best strategy for IDS (likely B or C)
```

---

### Week 12-14: Quantization-Aware Training & Full Pipeline Analysis

#### Deliverables
- INT8 quantized models (QAT)
- Complete compression pipeline evaluation
- Cross-dataset generalization under compression
- Phase 2 comprehensive report

#### Tasks

**6.1 Quantization-Aware Training** (Days 59-65)
```python
# compression/qat.py

class QATTrainer:
    """
    Quantization-Aware Training for INT8 deployment
    
    Process:
    1. Insert fake quantization nodes in graph
    2. Train for additional epochs (15-20)
    3. Calibrate with representative dataset (1000 samples)
    4. Convert to TFLite INT8
    5. Validate numerical accuracy
    """

# QAT configs:
qat_configs = [
    {
        "input_model": "best_pruned_30",  # 30% pruned
        "quantization": "int8",
        "epochs": 20,
        "learning_rate": 0.0001,  # Lower LR for QAT
    },
    {
        "input_model": "best_pruned_40",  # 40% pruned
        "quantization": "int8",
        "epochs": 20,
        "learning_rate": 0.0001,
    },
]
```

**6.2 Numerical Validation** (Days 66-68)
```
Script: scripts/09_qat_validation.py

Tests:
1. Output comparison (Float32 vs INT8)
   - Max absolute difference
   - Mean absolute error
   - Correlation coefficient
   
2. Per-attack accuracy comparison
   - Which attacks affected by quantization?
   - Precision loss patterns
   
3. Edge case handling
   - Min/max value clipping
   - Overflow detection

Output:
- reports/phase2/qat_numerical_validation.pdf
- Flagged attacks with >2% degradation
```

**6.3 Complete Pipeline Evaluation** (Days 69-75)
```
Script: scripts/10_full_pipeline_eval.py

Evaluation Matrix:
                     | Teacher | Student | Pruned  | Pruned | QAT    | QAT
                     |         | (KD)    | (30%)   | (40%)  | (30%)  | (40%)
---------------------|---------|---------|---------|--------|--------|--------
Binary Accuracy      | 98.5%   | 97.2%   | 96.8%   | 95.1%  | 96.5%  | 94.8%
Macro-F1             | 96.2%   | 94.5%   | 93.8%   | 91.2%  | 93.5%  | 90.9%
Model Size (KB)      | 2400    | 350     | 245     | 210    | 62     | 53
Inference (ms/CPU)   | 45      | 12      | 10      | 8      | 7.5    | 6.2
Parameters           | 850K    | 80K     | 56K     | 48K    | 56K    | 48K

Per-Attack Family:
Attack          | Teacher | KD    | Prune | QAT   | Degradation
----------------|---------|-------|-------|-------|-------------
DDoS            | 99.1%   | 98.5% | 98.0% | 97.8% | -1.3%
DoS             | 98.3%   | 97.1% | 96.5% | 96.0% | -2.3%
PortScan        | 96.5%   | 95.8% | 95.2% | 94.8% | -1.7%
Brute-Force     | 94.2%   | 92.5% | 90.1% | 88.9% | -5.3% ⚠️
Infiltration    | 89.1%   | 85.2% | 80.3% | 78.1% | -11.0% 🚨
Web-Attack      | 92.5%   | 90.1% | 87.8% | 86.2% | -6.3% ⚠️
...
```

**6.4 Cross-Dataset Generalization Analysis** (Days 76-78)
```
Critical Experiment: How does compression affect cross-dataset performance?

Test each compressed model on:
1. Same-dataset test set (NF-UQ)
2. Cross-dataset test set (NF-CSE)

Metrics:
- Accuracy gap (same vs cross)
- Per-attack transfer degradation
- Dataset artifact amplification under compression

Hypothesis: Compression amplifies dataset bias

Output:
- reports/phase2/cross_dataset_generalization.md
- plots/generalization_gap_by_compression.png
- Key finding: Quantify compression-bias interaction
```

**6.5 Phase 2 Comprehensive Report** (Days 79-84)
```
Document: reports/phase2_compression_analysis.md

Executive Summary:
- Compression pipeline results
- Attack-family vulnerability ranking
- Cross-dataset generalization findings
- Recommended final model

Detailed Sections:

1. Knowledge Distillation Analysis
   - Best KD configuration (T=?, alpha=?)
   - Per-attack KD efficiency
   - Hard sample analysis

2. Structured Pruning Analysis  
   - Optimal pruning ratio (30% vs 40%)
   - Attack degradation curves
   - Pruning strategy comparison

3. Quantization-Aware Training
   - Numerical accuracy analysis
   - Per-attack INT8 impact
   - Edge case handling

4. End-to-End Pipeline Performance
   - Compression stage comparison table
   - Pareto frontier (size vs accuracy)
   - Attack-family survival rates

5. Cross-Dataset Generalization
   - Same-dataset vs cross-dataset gaps
   - Compression amplifies bias: Evidence
   - Mitigation strategies (if any)

6. Attack Vulnerability Analysis
   - Robust attacks: Survive >40% compression
   - Vulnerable attacks: Degrade >5% at 30%
   - Catastrophic failures: >10% degradation

7. Recommendations
   - Selected model: Pruned_30_QAT or Pruned_40_QAT
   - Deployment confidence by attack type
   - Known limitations and failure modes
```

**DECISION GATE 3**: Final model must achieve:
- Binary accuracy ≥95% (same-dataset)
- Binary accuracy ≥92% (cross-dataset)  
- Model size ≤200KB (INT8)
- Inference time ≤15ms (Pi4 CPU, unoptimized)
- Major attacks (DDoS, DoS, PortScan) recall ≥95%

---

## 🚀 PHASE 3: DEPLOYMENT & EVALUATION (WEEKS 15-20)

**Goal**: Real hardware deployment, extensive benchmarking, paper writing

### Week 15-16: Raspberry Pi 4 Deployment & Benchmarking

#### Deliverables
- Deployed model on Pi4
- Comprehensive hardware benchmarks
- Real-world inference testing
- Optimization analysis

#### Tasks

**7.1 TFLite Model Conversion & Optimization** (Days 85-88)
```python
# deployment/tflite_converter.py

class TFLiteConverter:
    """
    Convert PyTorch/TF model to optimized TFLite
    
    Optimizations:
    - INT8 quantization (post-training or QAT)
    - Operator fusion
    - Constant folding
    - Select TensorFlow Lite operators
    
    Target: <200KB .tflite file
    """

# Conversion configs:
conversions = [
    {
        "model": "pruned_30_qat",
        "optimizations": ["int8"],
        "representative_dataset": "calibration_1000.npy"
    },
    {
        "model": "pruned_40_qat",
        "optimizations": ["int8"],
        "representative_dataset": "calibration_1000.npy"
    },
]

# Validation:
# - Load .tflite model
# - Run on test samples
# - Compare with original model outputs
# - Verify <0.5% accuracy loss
```

**7.2 Raspberry Pi 4 Setup** (Days 89-90)
```bash
# deployment/pi4_setup.sh

# Hardware: Raspberry Pi 4 Model B (4GB or 8GB RAM)
# OS: Raspberry Pi OS (64-bit, Debian-based)

# Install dependencies:
sudo apt-get update
sudo apt-get install python3.9 python3-pip
pip3 install tensorflow-lite numpy scipy

# Install monitoring tools:
sudo apt-get install sysstat htop

# Copy model and benchmark scripts:
scp models/final_compressed.tflite pi@raspberrypi:/home/pi/ids/
scp scripts/benchmark_pi4.py pi@raspberrypi:/home/pi/ids/
```

**7.3 Hardware Benchmarking Suite** (Days 91-94)
```python
# benchmarking/pi4_benchmark.py

class Pi4Benchmark:
    """
    Comprehensive hardware benchmarking on Raspberry Pi 4
    
    Metrics:
    1. Inference latency (single-thread CPU)
       - p50, p90, p95, p99 over 10,000 inferences
       - Warmup: 100 inferences (discard)
       
    2. Throughput
       - Flows per second (batch_size=1)
       - Flows per second (batch_size=8, 16, 32)
       
    3. Memory usage
       - Peak RAM consumption
       - Memory footprint (resident set size)
       
    4. CPU utilization
       - Average CPU % over 1000 inferences
       - Core usage distribution
       
    5. Temperature & Power
       - CPU temperature (°C) during sustained load
       - Estimated power consumption (if measurable)
       
    6. Multi-threading
       - 1, 2, 4 threads comparison
       - Optimal thread count
    """

# Benchmark scenarios:
scenarios = [
    {
        "name": "single_flow_latency",
        "batch_size": 1,
        "iterations": 10000,
        "threads": 1
    },
    {
        "name": "throughput_optimized",
        "batch_size": 32,
        "iterations": 1000,
        "threads": 4
    },
    {
        "name": "sustained_load",
        "batch_size": 1,
        "duration_seconds": 600,  # 10 minutes
        "threads": 2
    },
]
```

**7.4 Real-World Traffic Simulation** (Days 95-98)
```python
# benchmarking/traffic_simulation.py

class TrafficSimulator:
    """
    Simulate realistic network traffic patterns
    
    Scenarios:
    1. Normal traffic (90% benign, 10% attacks)
    2. Attack burst (sudden spike in malicious flows)
    3. Mixed traffic (varying benign/attack ratios)
    4. Long-running (24-hour simulation)
    """

# Traffic patterns:
patterns = [
    {
        "name": "normal_operation",
        "benign_rate": 100,  # flows/sec
        "attack_rate": 10,   # flows/sec
        "duration": 3600,    # 1 hour
    },
    {
        "name": "ddos_burst",
        "benign_rate": 50,
        "attack_rate": 500,
        "duration": 300,     # 5 minutes
        "attack_type": "DDoS"
    },
]

# Metrics:
# - Detection latency (time from flow to detection)
# - System stability (no crashes, memory leaks)
# - Accuracy under load
# - False positive rate in sustained operation
```

---

### Week 17-18: Ablation Studies & Comparative Analysis

#### Deliverables
- Complete ablation study results
- Comparison with related work
- Failure case analysis
- Limitations documentation

#### Tasks

**8.1 Comprehensive Ablation Studies** (Days 99-105)
```
Ablations to perform:

A. Dataset Fusion Impact
   - Single dataset (NF-UQ only) vs Merged (NF-UQ + NF-CSE)
   - Same-dataset test performance
   - Cross-dataset generalization
   
B. Compression Stage Impact  
   - Teacher → Student (KD)
   - Student → Pruned
   - Pruned → QAT
   - Which stage matters most?

C. Augmentation Strategy
   - No augmentation
   - Jitter only
   - Mixup only
   - Combined (jitter + mixup)
   - CTGAN (if used)

D. KD Configuration
   - Temperature: 2.0 vs 4.0 vs 6.0
   - Alpha: 0.5 vs 0.7 vs 0.9
   - Teacher ensemble vs single teacher

E. Pruning Strategy
   - Magnitude vs importance-based
   - Structured vs unstructured
   - Iterative vs one-shot

Output:
- reports/phase3/ablation_studies.md (with tables and plots)
- Quantify contribution of each component
```

**8.2 Comparison with Related Work** (Days 106-108)
```
Reproduce (if feasible) or cite recent baselines:

Baselines from 2024-2025:
1. DNN-KDQ (CICIDS2017, KD+Quantization, 20KB)
2. Ensemble-LSTM-Pi4 (IoT-23, Pruning+Quantization)
3. STKD (Various datasets, Spatial-temporal KD)

Comparison metrics:
- Model size (KB)
- Inference time (ms)
- Accuracy (binary and multi-class)
- Dataset used (normalize if possible)
- Deployment platform (Pi4 or similar)

Output:
- tables/comparison_with_sota.csv
- Fair comparison notes (different datasets, etc.)
- Our positioning: "We provide comprehensive attack-family analysis"
```

**8.3 Failure Case Analysis** (Days 109-112)
```
Deep dive into model failures:

1. Identify hard samples (misclassified by final model)
2. Categorize failure modes:
   - False positives (benign → attack)
   - False negatives (attack → benign)
   - Misclassification (attack A → attack B)

3. Per-attack-family failure analysis
   - Which attacks most confused?
   - Are failures correlated with dataset origin?
   - Do failures cluster in feature space?

4. Compression-induced failures
   - Which samples failed AFTER compression?
   - Are there patterns? (e.g., edge cases, rare features)

Output:
- reports/phase3/failure_analysis.md
- plots/failure_feature_space_tsne.png
- hard_samples_dataset.csv (for reproducibility)
```

---

### Week 19-20: Paper Writing & Reproducibility Package

#### Deliverables
- Complete research paper draft
- Reproducibility package (code + data + models)
- README and documentation
- Submission-ready materials

#### Tasks

**9.1 Paper Structure & Writing** (Days 113-120)
```
Paper: 8-10 pages (conference format: USENIX, NDSS, IEEE S&P, CCS)

Sections:

1. Abstract (250 words)
   - Problem: IDS compression for edge devices
   - Gap: Lack of attack-family compression analysis
   - Contribution: Systematic study + reproducible benchmark
   - Results: Achieved X accuracy at Y size with Z degradation pattern

2. Introduction (1.5 pages)
   - Motivation: Edge IDS deployment challenges
   - Problem statement: Compression affects attacks differently
   - Research questions:
     RQ1: How does compression affect different attack families?
     RQ2: Does dataset fusion improve robustness under compression?
     RQ3: What is the optimal compression-accuracy tradeoff?
   - Contributions (4 bullet points)

3. Related Work (1.5 pages)
   - IDS for edge devices
   - Model compression techniques (KD, pruning, quantization)
   - Multi-dataset IDS (dataset bias, fusion)
   - Gap: No systematic attack-family compression study

4. Methodology (2 pages)
   - Dataset preparation & fusion strategy
   - Baseline model selection (DS-1D-CNN)
   - Compression pipeline (Teacher → KD → Prune → QAT)
   - Evaluation protocol (same/cross-dataset, per-attack)

5. Experimental Setup (0.5 page)
   - Datasets: NF-UQ-NIDS-v2 + NF-CSE-CIC-IDS2018-v2
   - Hardware: Training (GPU) + Deployment (Pi4)
   - Implementation: TensorFlow/PyTorch, TFLite

6. Results (2.5 pages)
   - Overall compression results (table)
   - Per-attack-family degradation (plots)
   - Cross-dataset generalization under compression
   - Raspberry Pi 4 benchmarks (table)
   - Ablation study highlights

7. Analysis & Discussion (1 page)
   - Which attacks survive compression? (DDoS, DoS)
   - Which attacks vulnerable? (Infiltration, Web-Attack)
   - Why? (hypothesis: feature complexity, samples)
   - Dataset fusion impact: reduced bias amplification
   - Practical implications for deployment

8. Limitations (0.5 page)
   - Dataset limitations (simulated attacks)
   - Compression techniques explored (not exhaustive)
   - Single edge device (Pi4, not microcontrollers)
   - Attack coverage (20 types, not all variants)

9. Conclusion (0.5 page)
   - Summary of findings
   - Reproducibility package available
   - Future work: adaptive compression, online learning

10. References (1-2 pages)
    - 40-50 citations
```

**9.2 Reproducibility Package Creation** (Days 121-126)
```
GitHub Repository Structure:

ids-compression-benchmark/
│
├── README.md                  # Comprehensive setup guide
├── requirements.txt           # All dependencies
├── environment.yml            # Conda environment (optional)
│
├── configs/                   # All YAML configs used
├── data/                      # Data download scripts + samples
│   ├── download_datasets.sh
│   ├── README_data.md
│   └── sample_1000.parquet    # Small sample for testing
│
├── models/                    # Model architectures
│   ├── teacher.py
│   ├── ds_cnn.py
│   ├── mlp.py
│   └── lstm.py
│
├── compression/               # Compression modules
│   ├── knowledge_distillation.py
│   ├── structured_pruning.py
│   └── qat.py
│
├── benchmarking/              # Benchmarking scripts
│   ├── pi4_benchmark.py
│   ├── traffic_simulation.py
│   └── evaluate_all.py
│
├── scripts/                   # All experiment scripts (01-10)
│   ├── 01_data_inventory.py
│   ├── 02_canonical_schema.py
│   ├── ...
│   └── 10_full_pipeline_eval.py
│
├── experiments/               # Saved experiment results
│   ├── phase1_baselines/
│   ├── phase2_compression/
│   └── phase3_deployment/
│
├── reports/                   # Generated reports (Markdown + PDF)
├── plots/                     # All figures from paper
│
├── pretrained/                # Released models
│   ├── teacher_model/
│   ├── final_compressed_30.tflite
│   ├── final_compressed_40.tflite
│   └── model_cards.md
│
├── tests/                     # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   └── test_compression.py
│
└── paper/                     # LaTeX source (if applicable)
    ├── main.tex
    ├── figures/
    └── compile.sh

Key Files:

README.md:
- Quick start (5 minutes)
- Full reproduction (step-by-step, estimated 2 weeks)
- Hardware requirements
- Expected outputs
- Citation
- License

requirements.txt:
tensorflow==2.x.x
torch==2.x.x
numpy==1.x.x
pandas==2.x.x
scikit-learn==1.x.x
matplotlib==3.x.x
seaborn==0.x.x
tqdm==4.x.x
pyyaml==6.x.x
...

LICENSE:
- MIT or Apache 2.0 (open source)
```

**9.3 Model Cards & Documentation** (Days 127-130)
```
Create model cards for each released model:

models/model_cards/final_compressed_30.md:
---
Model: DS-1D-CNN (30% Pruned + INT8 QAT)
Size: 62 KB
Inference: 7.5 ms (Pi4 CPU, single-thread)
Trained on: NF-UQ-NIDS-v2 (primary) + NF-CSE-CIC-IDS2018-v2

Performance:
- Binary Accuracy: 96.5% (same-dataset), 93.8% (cross-dataset)
- Macro-F1: 93.5%

Per-Attack Performance:
DDoS:        Precision: 97.8%, Recall: 98.1%, F1: 97.9%
DoS:         Precision: 96.2%, Recall: 95.8%, F1: 96.0%
PortScan:    Precision: 94.9%, Recall: 94.7%, F1: 94.8%
Brute-Force: Precision: 89.5%, Recall: 88.2%, F1: 88.9%
...

Known Limitations:
- Reduced recall on Infiltration attacks (78.1%, down from 89.1%)
- Higher false positives on Web-Attack (6.3% degradation)
- Cross-dataset accuracy gap: 2.7%

Recommended Use:
- High-volume traffic (DDoS, DoS, PortScan): Excellent
- Balanced accuracy/speed: Good
- Rare attacks (Infiltration): Use with caution

Not Recommended:
- Safety-critical deployments requiring >95% recall on all attacks
- Zero-tolerance false positive environments

Ethical Considerations:
- May miss sophisticated infiltration attempts
- Should be part of defense-in-depth strategy

Training Details:
- Compression stages: KD (T=4.0, α=0.7) → Prune (30%, L1×grad) → QAT (20 epochs)
- Training time: ~48 hours (GPU)
- Carbon footprint: ~X kg CO2eq

Citation:
[Your paper citation]
---
```

**9.4 Final Validation & Submission Prep** (Days 131-140)
```
Checklist before submission:

Code Quality:
✓ All scripts run without errors
✓ Tests pass (pytest)
✓ Linting (flake8/pylint)
✓ Type hints (mypy)
✓ Docstrings complete

Reproducibility:
✓ Requirements.txt freezes versions
✓ README step-by-step verified
✓ Sample dataset runs in <30 min
✓ Pretrained models loadable
✓ Figures reproducible from code

Paper Quality:
✓ All claims supported by experiments
✓ All figures have captions
✓ All tables have captions
✓ References formatted correctly
✓ No orphaned citations
✓ Spell-checked
✓ Grammar-checked (Grammarly/LanguageTool)
✓ Limitations section honest and complete
✓ Ethical considerations addressed
✓ Reproducibility statement included

Supplementary Materials:
✓ Extended results (appendix)
✓ Hyperparameter search logs
✓ Full confusion matrices
✓ All ablation study results

Submission Package:
├── paper.pdf
├── supplementary.pdf
├── code_and_data.zip (or GitHub link)
├── model_card.pdf
└── ethical_considerations_statement.pdf

Target Venues (ranked by fit):
1. USENIX Security (Deadline: February/August)
2. NDSS (Deadline: ~May)
3. IEEE Symposium on Security and Privacy (Deadline: ~May)
4. ACM CCS (Deadline: ~May)
5. RAID (Deadline: ~March)

Backup Venues (if rejected):
- IEEE INFOCOM (networking focus)
- ICICS (applied security)
- Computer Networks Journal (longer form)
```

---

## 📊 SUCCESS CRITERIA & MILESTONES

### Phase 1 Success Criteria
- ✅ Datasets downloaded and validated (>99% intact)
- ✅ Canonical schema implemented (43 features aligned)
- ✅ Baseline models achieve: Binary acc ≥97%, Macro-F1 ≥90%
- ✅ Decision report documents go/no-go with evidence

### Phase 2 Success Criteria  
- ✅ Teacher model outperforms baselines by ≥2% macro-F1
- ✅ KD achieves ≥95% of teacher performance at 10× smaller size
- ✅ Structured pruning reaches 30% without >5% macro-F1 drop
- ✅ QAT maintains ≥98% of pruned model accuracy
- ✅ Per-attack degradation curves generated for all 20 classes
- ✅ Cross-dataset generalization gap quantified (<5% acceptable)

### Phase 3 Success Criteria
- ✅ Model runs on Pi4 with ≤15ms latency (single-flow)
- ✅ Model size ≤200KB (INT8 TFLite)
- ✅ Sustained operation (10+ minutes) without crashes
- ✅ Ablation studies complete (5 major ablations)
- ✅ Paper draft complete (8+ pages)
- ✅ Reproducibility package tested by external person

### Publication Success Metrics
**Minimum Viable Publication**:
- Compressed model: ≤200KB, ≤15ms inference
- Binary accuracy: ≥95% (same), ≥92% (cross)
- Major attacks recall: ≥95%
- Novel contribution: Per-attack compression analysis
- Reproducibility package: Complete

**Strong Publication**:
- Compressed model: ≤100KB, ≤10ms inference
- Binary accuracy: ≥96% (same), ≥94% (cross)
- All attacks recall: ≥90% (except 2-3 documented)
- Dataset fusion reduces bias: Demonstrated quantitatively
- Multiple ablations: All completed
- Comparison with SOTA: Favorable or competitive

**Exceptional Publication (Top-tier venue)**:
- Compressed model: ≤80KB, ≤8ms inference
- Binary accuracy: ≥97% (same), ≥95% (cross)
- Novel insights: Compression-attack relationship theory
- Real deployment: 24+ hour stability testing
- Industry impact: Cited use case or adoption

---

## ⚠️ RISK MANAGEMENT & MITIGATION

### Timeline Risks

**Risk**: Phase 2 pruning takes longer than 3 weeks  
**Mitigation**: Start with coarse-grained pruning (10%, 20%, 30% only). Fine-grained can be follow-up work.  
**Fallback**: Skip 40-50% pruning if 30% meets publication criteria.

**Risk**: Pi4 hardware unavailable or broken  
**Mitigation**: Order backup Pi4, or use CPU benchmarking on workstation with scaling factor.  
**Fallback**: Simulate Pi4 performance using published benchmarks.

**Risk**: Cross-dataset performance gap >10%  
**Mitigation**: Apply domain adaptation techniques (adversarial loss, CORAL).  
**Fallback**: Frame as "compression amplifies bias" - still publishable finding.

**Risk**: Paper rejected at top venue  
**Mitigation**: Target 3-4 backup venues from day 1. Have resubmission plan.  
**Fallback**: Extend work with additional datasets or techniques for resubmission.

### Technical Risks

**Risk**: KD doesn't improve student performance  
**Mitigation**: Extensive hyperparameter search (temperature, alpha). Try ensemble teachers.  
**Fallback**: Publish "KD limited for IDS" - negative results are valuable.

**Risk**: Structured pruning causes catastrophic failure  
**Mitigation**: Iterative pruning with fine-tuning. Stop at first significant drop.  
**Fallback**: Use unstructured pruning (less efficient but safer).

**Risk**: QAT introduces severe quantization errors  
**Mitigation**: Extensive calibration dataset. Lower learning rate. More epochs.  
**Fallback**: Use post-training quantization (slightly lower accuracy, but works).

**Risk**: Rare attacks completely fail under compression  
**Mitigation**: This is acceptable if documented transparently. It's a limitation, not a failure.  
**Fallback**: Focus paper on "attacks that survive compression" as primary contribution.

### Data Risks

**Risk**: Dataset download fails or corrupted  
**Mitigation**: Multiple download sources. Checksum verification. Contact authors directly.  
**Fallback**: Use publicly available alternatives (CIC-IDS2017, UNSW-NB15).

**Risk**: Feature alignment between datasets impossible  
**Mitigation**: Canonical schema designed for flexibility. Use imputation for missing features.  
**Fallback**: Train separate models per dataset, compare results (different contribution angle).

**Risk**: Dataset bias too severe to merge  
**Mitigation**: Apply domain adaptation techniques. Extensive bias analysis in paper.  
**Fallback**: Frame as "dataset bias study under compression" - equally valuable.

---

## 📚 WEEKLY PROGRESS TRACKING

Use this template for weekly reports:

```markdown
# Week X Progress Report (Dates)

## Completed Tasks
- [x] Task 1 with details
- [x] Task 2 with details
- [ ] Task 3 (partial)

## Key Results
- Metric 1: X% (target: Y%)
- Metric 2: Z units (target: W units)

## Blockers & Issues
1. Issue: Description
   - Root cause: Analysis
   - Resolution plan: Steps
   - ETA: Date

## Next Week Plan
- [ ] Task A (priority: high)
- [ ] Task B (priority: medium)
- [ ] Task C (priority: low)

## Timeline Status
- On track / At risk / Behind
- If behind: Recovery plan

## Decision Points
- Decision needed: Yes/No
- What: Description
- By when: Date
```

---

## 🎓 PUBLICATION STRATEGY

### Target Venues (Ranked)

**Tier 1 (Ideal Fit)**:
1. **USENIX Security** - Strong ML security track, values reproducibility
2. **NDSS** - Network security focus, accepts compression work
3. **IEEE S&P** - Accepts applied ML work, high impact

**Tier 2 (Good Fit)**:
4. **ACM CCS** - Broader security, competitive but fair
5. **RAID** - Intrusion detection focus, very relevant

**Tier 3 (Backup)**:
6. **IEEE INFOCOM** - Networking + ML, less competitive
7. **Computer Networks Journal** - Longer paper format
8. **IEEE Access** - Open access, faster turnaround

### Submission Timeline

**If starting Week 1 on January 1:**
- Week 20 complete: ~May 20
- Paper draft ready: ~May 27
- Submission deadline target: August/September conferences
- Realistic: USENIX Security (August deadline) or NDSS (May deadline next cycle)

**Key Dates to Watch**:
- USENIX Security: ~February (summer) or ~August (winter)
- NDSS: ~May (annual)
- IEEE S&P: ~May (first cycle), ~August (second cycle)
- CCS: ~May (annual)

### Submission Checklist

**Before Submission**:
- [ ] Title captures contribution (not "Compressing..." but "Systematic Analysis of...")
- [ ] Abstract <250 words, self-contained
- [ ] Contributions clearly stated (3-4 bullets)
- [ ] All figures have vector graphics (PDF/EPS)
- [ ] All tables formatted consistently
- [ ] Limitations section: Honest and complete
- [ ] Reproducibility: Code/data available (link in paper)
- [ ] Ethical considerations: Addressed
- [ ] Author contributions: Documented
- [ ] Funding/conflict of interest: Disclosed
- [ ] References: 40-50 citations, recent (50% from 2020-2025)

**After Submission**:
- [ ] Upload to arXiv (optional, check venue policy)
- [ ] Release code on GitHub (public or private initially)
- [ ] Prepare rebuttal template
- [ ] Monitor submission system

---

## 🔧 TOOLS & INFRASTRUCTURE

### Required Software
- Python 3.9-3.11
- TensorFlow 2.13+ or PyTorch 2.0+
- TFLite runtime
- scikit-learn, pandas, numpy
- matplotlib, seaborn (visualization)
- tqdm (progress bars)
- wandb or tensorboard (experiment tracking)

### Optional but Recommended
- Jupyter notebooks (exploratory analysis)
- DVC (data version control)
- MLflow (experiment tracking)
- Docker (reproducibility)

### Hardware Requirements

**Development/Training**:
- GPU: NVIDIA RTX 3080+ or equivalent (12GB+ VRAM)
- RAM: 32GB+ (for large datasets)
- Storage: 500GB+ SSD (datasets + experiments)
- CPU: 8+ cores (for data preprocessing)

**Deployment/Testing**:
- Raspberry Pi 4 Model B (4GB or 8GB)
- MicroSD card: 32GB+ (class 10)
- Power supply: Official 5V/3A
- Cooling: Heatsink or fan (for sustained load)

### Compute Budget Estimate

**Training**:
- Teacher model: ~20 GPU hours
- Student models (KD): ~30 GPU hours
- Pruning experiments: ~10 GPU hours
- QAT: ~15 GPU hours
- Ablations: ~25 GPU hours
- **Total: ~100 GPU hours** (or ~$200-300 on cloud GPU)

**Alternatives**:
- Google Colab Pro: $10/month (limited GPU hours)
- AWS EC2 p3.2xlarge: ~$3/hour
- University/lab GPU cluster: Free
- Kaggle Notebooks: 30 hours/week free GPU

---

## 📖 RECOMMENDED READING

### Core Papers (Must Read)

**Model Compression**:
1. Hinton et al. "Distilling the Knowledge in a Neural Network" (2015) - Foundational KD
2. Han et al. "Deep Compression" (2016) - Pruning + Quantization
3. Howard et al. "MobileNets" (2017) - Depthwise separable convolutions

**Intrusion Detection**:
4. Sharafaldin et al. "Toward Generating a New Intrusion Detection Dataset" (2018) - CIC-IDS2018
5. Sarhan et al. "NetFlow Datasets for Machine Learning" (2021) - NF datasets overview
6. Recent: Any 2024-2025 paper on IDS compression for edge devices

**Cross-Dataset Generalization**:
7. Arp et al. "Dos and Don'ts of ML in Computer Security" (2022) - Dataset bias
8. Any recent "dataset shift" or "domain adaptation for IDS" papers

### Supplementary Reading

- Quantization-aware training guides (TensorFlow/PyTorch docs)
- TFLite optimization best practices
- Raspberry Pi performance tuning guides

---

## 💡 EXPECTED CONTRIBUTIONS (For Paper)

### Main Contributions

**C1: Systematic Per-Attack-Family Compression Analysis**
> We present the first comprehensive study of how multi-stage compression (knowledge distillation, structured pruning, quantization) affects different attack families in network intrusion detection, revealing that high-volume attacks (DDoS, DoS) maintain >95% recall even at 40% pruning, while sophisticated attacks (Infiltration, Web-Attack) degrade significantly.

**C2: Cross-Dataset Generalization Under Compression**
> We quantify the interaction between dataset fusion and model compression, demonstrating that merging diverse NetFlow datasets reduces cross-dataset accuracy gaps by X% compared to single-dataset training, but compression amplifies dataset bias by Y%, requiring explicit mitigation strategies.

**C3: Reproducible Compression Benchmark Suite**
> We release a complete open-source pipeline including dataset fusion, provenance tracking, compression stages, and Raspberry Pi 4 deployment scripts, enabling future researchers to reproduce our results and extend compression studies to new attack types or architectures.

**C4: Deployment-Ready Compressed Model**
> We provide production-ready INT8 models (62KB and 53KB) achieving 96.5%/94.8% binary accuracy with 7.5ms/6.2ms inference latency on Raspberry Pi 4, suitable for real-world edge IDS deployment with documented performance characteristics per attack family.

### Minor Contributions (Mention in Paper)

- Canonical NetFlow schema for multi-dataset research
- Importance-based structured pruning strategy for 1D-CNN
- Per-attack hard sample analysis methodology
- Extended benchmarking suite for edge IDS evaluation

---

## 🚨 RED FLAGS TO AVOID (Reviewer Concerns)

### Common Rejection Reasons

**1. Insufficient Novelty**
> ❌ "We compressed a model" ← This is not enough  
> ✅ "We systematically analyzed which attacks survive compression" ← Novel insight

**2. Lack of Rigor**
> ❌ Single dataset, single run, cherry-picked results  
> ✅ Multiple datasets, seeded runs (n=5), complete ablations

**3. Weak Evaluation**
> ❌ Only overall accuracy reported  
> ✅ Per-class metrics, cross-dataset, confusion matrices, ROC curves

**4. Overstated Claims**
> ❌ "Our model is the smallest/fastest ever"  
> ✅ "Our model achieves X accuracy at Y size, with documented tradeoffs"

**5. Poor Reproducibility**
> ❌ No code, vague hyperparameters, missing datasets  
> ✅ Complete code, exact configs, downloadable datasets, pretrained models

**6. Ignored Limitations**
> ❌ Only presenting positive results  
> ✅ Honest limitations section, failure case analysis

### How to Address in Paper

**Novelty Defense** (in Introduction):
> "While model compression for IDS has been explored [cite 3-4 recent works], prior work focuses on overall accuracy metrics. We provide the first systematic per-attack-family analysis, revealing heterogeneous compression impacts..."

**Rigor Defense** (in Methodology):
> "All experiments are run with 5 random seeds. We report mean ± std. We use stratified splits to ensure class balance. We hold out an entire dataset for cross-dataset evaluation..."

**Limitation Acknowledgment** (in Limitations):
> "Our study is limited to 20 attack types from two NetFlow datasets. Real-world attacks may exhibit different characteristics. Our compression techniques (KD, pruning, QAT) are not exhaustive; other methods (e.g., neural architecture search, lottery ticket) may yield different results..."

---

## 📝 FINAL THOUGHTS & SUCCESS FACTORS

### What Will Make This Work Succeed

1. **Honesty**: Don't oversell. Reviewers value transparency.
2. **Thoroughness**: Complete ablations beat clever tricks.
3. **Reproducibility**: Code + data + models = credibility.
4. **Focus**: Deep analysis beats broad shallow coverage.
5. **Realistic Targets**: Achievable goals beat ambitious failures.

### What Will Make This Work Fail

1. **Rushing**: Cutting corners to meet deadlines.
2. **Cherry-picking**: Only reporting best results.
3. **Vague Claims**: "Our model is efficient" without numbers.
4. **Ignoring Failures**: Not analyzing why things didn't work.
5. **Poor Writing**: Good work with bad presentation fails.

### Timeline Realism

**This is a 20-week (5-month) project if done right.**
- Phases 1-2: Research and development (12-14 weeks)
- Phase 3: Deployment and writing (6-8 weeks)
- Buffer: 2-4 weeks for unexpected issues

**Attempting this in 12-14 weeks will force compromises:**
- Fewer ablations
- Less thorough evaluation
- Rushed writing
- Higher risk of rejection

**My honest recommendation**: Take 20 weeks. The difference between a rejected paper and an accepted one is often thoroughness, not brilliance.

---

## 🎯 FINAL RECOMMENDATION

**Your dataset choice is excellent. Your approach is sound. Your plan was rushed.**

**Do this:**
1. Follow this 20-week plan (not 12-14 weeks)
2. Focus on per-attack compression analysis (your differentiator)
3. Be brutally honest about limitations
4. Release everything (code, data, models)
5. Target USENIX Security or NDSS (good fit)

**Avoid this:**
1. Don't claim "smallest model ever" - you won't be
2. Don't skip cross-dataset evaluation - critical for credibility
3. Don't rush ablations - reviewers will notice
4. Don't hide failures - they're part of the story

**Your competitive advantage**:
- Systematic approach (not ad-hoc)
- Reproducibility focus (many papers lack this)
- Per-attack analysis (nobody has done this thoroughly)
- Honest limitations (builds trust)

**Your publication probability** (if you follow this plan):
- Top-tier venue (USENIX, NDSS, S&P): 40-50%
- Second-tier venue (CCS, RAID): 70-80%
- Journal (Computer Networks, IEEE Access): 90%+

**Make it happen. You've got this. 🚀**