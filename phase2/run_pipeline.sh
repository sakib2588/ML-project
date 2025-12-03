#!/bin/bash
set -e

# ============================================================
# Phase 2 Compression Pipeline Runner
# ============================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
SEEDS="0,7,21,42,101,202,303"
ALL_SEEDS=(0 7 21 42 101 202 303)
DEFAULT_SEED=42
START_STAGE=0
END_STAGE=6
DEVICE="cpu"
SKIP_TEACHER=false
DRY_RUN=false
QUICK_MODE=false
SKIP_PRE_CHECK=false
FROM_STAGE=""
SINGLE_STAGE=""
SINGLE_SEED=""

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PHASE2_DIR="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv_edge"
ARTIFACTS_DIR="${PHASE2_DIR}/artifacts"

# Config
PRUNE_RATIO=0.5
N_PRUNE_STEPS=10
QUANTIZE="int8"
SCHEDULE="iterative_50"

print_help() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 0
}

# Arg parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)        SINGLE_SEED="$2"; shift 2 ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        --stage)       SINGLE_STAGE="$2"; shift 2 ;;
        --from)        FROM_STAGE="$2"; shift 2 ;;
        --to)          END_STAGE="$2"; shift 2 ;;
        --skip-teacher) SKIP_TEACHER=true; shift ;;
        --device)      DEVICE="$2"; shift 2 ;;
        --quick)       QUICK_MODE=true; shift ;;
        --skip-pre-check) SKIP_PRE_CHECK=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)     print_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

if [[ -n "$SINGLE_SEED" ]]; then
    SEEDS_TO_RUN=("$SINGLE_SEED")
elif [[ "$QUICK_MODE" == true ]]; then
    SEEDS_TO_RUN=("$DEFAULT_SEED")
else
    SEEDS_TO_RUN=("${ALL_SEEDS[@]}")
fi

# Virtual env
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        echo "[RUN] $*"
        eval "$@"
    fi
}

train_teacher() {
    echo "== Teacher Training =="
    TEACHER_PATH="${ARTIFACTS_DIR}/teacher/best_teacher.pt"
    if [ -f "$TEACHER_PATH" ] && [ "$SKIP_TEACHER" = true ]; then
        echo "Teacher exists, skipping."
        return
    fi
    run_cmd "python ${PHASE2_DIR}/train/train_kd.py --train-teacher --epochs 100 --device $DEVICE"
    [ -f "$TEACHER_PATH" ] || { echo "Teacher training failed"; exit 1; }
}

run_stage0() {
    echo "== Stage 0: Baseline =="
    for seed in "${SEED_ARRAY[@]}"; do
        run_cmd "python ${PHASE2_DIR}/train/train_baseline.py --seed $seed --epochs 100 --device $DEVICE"
    done
}

run_stage1() {
    echo "== Stage 1: KD =="
    train_teacher
    for seed in "${SEED_ARRAY[@]}"; do
        run_cmd "python ${PHASE2_DIR}/train/train_kd.py --seed $seed --epochs 100 --device $DEVICE"
    done
}

run_stage2() {
    echo "== Stage 2: Pruning =="
    for seed in "${SEED_ARRAY[@]}"; do
        MODEL_PATH="${ARTIFACTS_DIR}/stage1/seed${seed}/best_model.pt"
        [ -f "$MODEL_PATH" ] || { echo "Missing model $MODEL_PATH"; continue; }
        run_cmd "python ${PHASE2_DIR}/prune/prune_model.py --model $MODEL_PATH --seed $seed --mode iterative --ratio $PRUNE_RATIO --n-steps $N_PRUNE_STEPS --device $DEVICE"
    done
}

run_stage3() {
    echo "== Stage 3: KD Fine-tune =="
    for seed in "${SEED_ARRAY[@]}"; do
        MODEL_PATH="${ARTIFACTS_DIR}/stage2/seed${seed}/pruned_model.pt"
        [ -f "$MODEL_PATH" ] || { echo "Missing pruned model $MODEL_PATH"; continue; }
        run_cmd "python ${PHASE2_DIR}/prune/fine_tune_kd.py --model $MODEL_PATH --seed $seed --epochs 50 --device $DEVICE"
    done
}

run_stage4() {
    echo "== Stage 4: QAT =="
    for seed in "${SEED_ARRAY[@]}"; do
        MODEL_PATH="${ARTIFACTS_DIR}/stage3/seed${seed}/best_model.pt"
        [ -f "$MODEL_PATH" ] || { echo "Missing model $MODEL_PATH"; continue; }
        run_cmd "python ${PHASE2_DIR}/quant/qat_train.py --model $MODEL_PATH --seed $seed --epochs 30 --device $DEVICE"
    done
}

run_stage5() {
    echo "== Stage 5: TFLite =="
    for seed in "${SEED_ARRAY[@]}"; do
        MODEL_PATH="${ARTIFACTS_DIR}/stage4/seed${seed}/qat_model.pt"
        [ -f "$MODEL_PATH" ] || { echo "Missing QAT model $MODEL_PATH"; continue; }
        run_cmd "python ${PHASE2_DIR}/convert/convert_to_tflite.py --model $MODEL_PATH --seed $seed --optimize --quantize int8"
    done
}

run_stage6() {
    echo "== Stage 6: Analysis =="
    run_cmd "python ${PHASE2_DIR}/analysis/analyze_results.py --full"
}

run_pre_check() {
    echo "== Pre-Phase-2 Data Check =="
    run_cmd "python ${PHASE2_DIR}/pre_phase2/check_data_distribution.py" || return 1
}

run_augment() {
    echo "== Augment Rare Classes =="
    run_cmd "python ${PHASE2_DIR}/pre_phase2/augment_rare_classes.py --target-classes Bot,SSH-Patator --target-samples 5000 --method smote --seed ${SEEDS_TO_RUN[0]}"
}

should_run_from() {
    local stage=$1
    local order=(stage0 stage1 stage2 stage3 stage4 stage5 stage6)
    [ -z "$FROM_STAGE" ] && return 0
    local from_idx=-1 stage_idx=-1
    for i in "${!order[@]}"; do
        [[ "${order[$i]}" == "$FROM_STAGE" ]] && from_idx=$i
        [[ "${order[$i]}" == "$stage" ]] && stage_idx=$i
    done
    [[ $stage_idx -ge $from_idx ]]
}

# Single stage override
if [[ -n "$SINGLE_STAGE" ]]; then
    case "$SINGLE_STAGE" in
        pre_check) run_pre_check ;;
        stage0)    run_stage0 ;;
        stage1)    run_stage1 ;;
        stage2)    run_stage2 ;;
        stage3)    run_stage3 ;;
        stage4)    run_stage4 ;;
        stage5)    run_stage5 ;;
        stage6|analysis) run_stage6 ;;
        *) echo "Unknown stage: $SINGLE_STAGE"; exit 1 ;;
    esac
    exit 0
fi

# Pre-check
if [[ "$SKIP_PRE_CHECK" != true ]]; then
    if ! run_pre_check; then
        run_augment
    fi
fi

# Full pipeline
for n in $(seq $START_STAGE $END_STAGE); do
    case $n in
        0) stage_name=stage0; should_run_from stage0 && run_stage0 ;;
        1) stage_name=stage1; should_run_from stage1 && run_stage1 ;;
        2) stage_name=stage2; should_run_from stage2 && run_stage2 ;;
        3) stage_name=stage3; should_run_from stage3 && run_stage3 ;;
        4) stage_name=stage4; should_run_from stage4 && run_stage4 ;;
        5) stage_name=stage5; should_run_from stage5 && run_stage5 ;;
        6) stage_name=stage6; should_run_from stage6 && run_stage6 ;;
    esac
done

echo "Pipeline complete."
echo "Artifacts: ${ARTIFACTS_DIR}"
