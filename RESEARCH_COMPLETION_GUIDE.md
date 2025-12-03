# Research Completion & Publication Roadmap: Edge-Native Multi-Task IDS

This document outlines the step-by-step plan to complete the research project, ensuring the results are robust, reproducible, and suitable for high-impact publication (e.g., IEEE IoT Journal, DATE, ICC).

## 1. Research Objective
**Goal:** Develop a highly efficient, multi-task Intrusion Detection System (IDS) capable of running on resource-constrained edge devices (Raspberry Pi 4) without compromising detection accuracy for critical attacks (DDoS, PortScan).

**Key Contribution:** Demonstrating that **Knowledge Distillation (KD)** combined with **Quantization-Aware Training (QAT)** allows a tiny model (<100KB) to match the performance of a heavy server-grade model on complex multi-class intrusion detection tasks.

---

## 2. Current Status (Phase 2)
*   **Teacher Model (Phase 1):** Complete. High accuracy, but computationally heavy.
*   **Student Baselines (Stage 0):**
    *   ✅ **5K Params:** Complete (Acc ~96.5%).
    *   ✅ **50K Params:** Complete (Acc ~97.2%).
    *   ⏳ **200K Params:** Pending (Script `run_200k_sweep.py` ready).

---

## 3. Execution Roadmap (The "How-To")

### Stage 0: Complete the Baselines (Immediate Priority)
*   **Action:** Run the 200K parameter sweep.
*   **Command:** `nohup python run_200k_sweep.py > logs/200k_sweep.log 2>&1 &`
*   **Why:** We need a "control group". We must prove that the small models *cannot* reach peak performance on their own, justifying the need for Knowledge Distillation.

### Stage 1: Knowledge Distillation (KD)
*   **Action:** Train the 5K, 50K, and 200K models again, but this time using the **Phase 1 Teacher** to guide them.
*   **Mechanism:** The student learns from both the "Hard Labels" (Ground Truth) and "Soft Labels" (Teacher's probability distribution).
*   **Hypothesis:** KD Students will have higher Recall on rare classes (WebAttack, Bot) compared to Baseline Students.
*   **Script:** `src/training/train_kd.py` (Already created, needs execution).

### Stage 2: Optimization (Pruning & QAT)
*   **Action:** Take the best performing KD Student (likely 50K or 200K) and compress it further.
*   **Step A (Pruning):** Remove redundant neurons.
*   **Step B (QAT):** Retrain the model using simulated 8-bit integers (Int8) instead of 32-bit floats (FP32).
*   **Why:** This ensures the model runs fast on the Raspberry Pi's CPU/NPU.

### Stage 3: Edge Deployment & Benchmarking
*   **Action:** Convert the final model to TensorFlow Lite (`.tflite`).
*   **Action:** Run `benchmark_pi4.py` on the target hardware.
*   **Metrics:** Measure Latency (ms), Throughput (inferences/sec), and RAM usage.

---

## 4. Publication-Ready Results (The "What It Looks Like")

To be accepted into a top-tier venue, your results section must contain the following artifacts:

### A. Tables

**Table 1: Architecture & Efficiency Comparison**
| Model | Params | Size (MB) | FLOPs | Latency (Pi 4) |
|-------|--------|-----------|-------|----------------|
| Teacher (Baseline) | ~1M | 4.0 | ~10M | ~15ms |
| Student (50K) | 50K | 0.2 | ~0.5M | ~1ms |
| **Student (5K)** | **5K** | **0.02** | **~0.05M** | **<0.5ms** |

**Table 2: Detection Performance (Macro-Average)**
| Method | Accuracy | Precision | Recall | F1-Score | FAR |
|--------|----------|-----------|--------|----------|-----|
| Teacher | 99.1% | 99.0% | 98.8% | 98.9% | 0.5% |
| Student (Baseline) | 96.5% | 95.0% | 94.0% | 94.5% | 2.1% |
| **Student (KD + QAT)** | **98.5%** | **98.2%** | **98.0%** | **98.1%** | **0.8%** |

*Note: The "Student (KD)" should be much closer to the Teacher than the "Student (Baseline)".*

**Table 3: Critical Attack Granularity**
| Attack Type | Teacher Recall | Student (Base) Recall | Student (KD) Recall |
|-------------|----------------|-----------------------|---------------------|
| DDoS | 99.5% | 96.0% | **98.5%** |
| PortScan | 99.8% | 97.5% | **99.0%** |
| Web Attack | 85.0% | 60.0% | **80.0%** |

### B. Visualizations

1.  **The Pareto Frontier Plot:**
    *   **X-Axis:** Latency (ms) [Log Scale]
    *   **Y-Axis:** F1-Score
    *   **Content:** Plot points for Teacher, Baseline Students, and KD Students.
    *   **Goal:** Show that your KD models push the boundary (high accuracy, low latency) compared to standard training.

2.  **Confusion Matrices:**
    *   Side-by-side comparison: **Teacher** vs. **Best Student**.
    *   Highlight that the Student does not confuse "Benign" with "Attack" (Low False Positives).

3.  **Training Stability:**
    *   Loss curves showing KD converges faster or more stably than baseline training.

---

## 5. Robustness Checklist (Before Submitting)

- [ ] **Reproducibility:** Run every experiment 3 times (Seeds 0, 7, 42) and report Mean ± Std Dev.
- [ ] **Ablation Study:** Prove that the "Multi-Task" head is better than a simple binary classifier. (Does learning specific attacks help detect attacks in general?)
- [ ] **Hardware Validation:** Ensure latency numbers are from real hardware (or a high-fidelity simulator), not just theoretical calculations.
- [ ] **Comparison:** Compare against 1-2 other lightweight methods (e.g., MobileNet, SqueezeNet) if possible, or cite them as baselines.

## 6. Next Immediate Step
Execute the 200K sweep to complete Stage 0.

```bash
nohup python run_200k_sweep.py > logs/200k_sweep.log 2>&1 &
```
