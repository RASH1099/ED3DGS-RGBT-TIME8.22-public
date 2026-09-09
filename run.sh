#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${ED3DGS_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
SCENE=${ED3DGS_SCENE:-covers}
ACTION=${1:-help}
DATASET=${ED3DGS_DATASET:-}
RUN_ROOT=${ED3DGS_OUTPUT_ROOT:-$ROOT/outputs/runs}
TEACHER=${ED3DGS_TEACHER_MODEL_PATH:-$ROOT/outputs/pretrained/$SCENE/rgb_teacher}
if [[ -n "${ED3DGS_CONFIG:-}" ]]; then
  CONFIG=$ED3DGS_CONFIG
elif [[ -f "$ROOT/arguments/${SCENE}.py" ]]; then
  CONFIG="$ROOT/arguments/${SCENE}.py"
else
  CONFIG="$ROOT/arguments/MeetingRoom.py"
fi
if [[ -n "${ED3DGS_TEACHER_CONFIG:-}" ]]; then
  TEACHER_CONFIG=$ED3DGS_TEACHER_CONFIG
elif [[ -f "$ROOT/arguments/${SCENE}_rgb_teacher.py" ]]; then
  TEACHER_CONFIG="$ROOT/arguments/${SCENE}_rgb_teacher.py"
else
  TEACHER_CONFIG="$ROOT/arguments/MeetingRoom_rgb_teacher.py"
fi
DEFAULT_GT_LOCK_MODEL="$TEACHER"
GT_LOCK_MODEL=${ED3DGS_GT_LOCK_MODEL:-$DEFAULT_GT_LOCK_MODEL}
BASELINE_GATE=${ED3DGS_BASELINE_GATE_REPORT:-}
GATE_ITERATIONS=${ED3DGS_GATE_ITERATIONS:-1000}
case "$GATE_ITERATIONS" in
  64|1000|2560) ;;
  *) echo "Gate iterations must be 64, 1000, or 2560" >&2; exit 3 ;;
esac
STRICT_SCENE_FREEZE=${ED3DGS_STRICT_SCENE_FREEZE_V1:-1}
case "$STRICT_SCENE_FREEZE" in
  0|1) ;;
  *) echo "ED3DGS_STRICT_SCENE_FREEZE_V1 must be 0 or 1" >&2; exit 3 ;;
esac
ARM=${ED3DGS_EXPERIMENT_ARM:-full}
case "$ARM" in
  baseline) ABLATION_MODE=; STRICT_SCENE_FREEZE=0 ;;
  full) ABLATION_MODE= ;;
  time_only) ABLATION_MODE=frozen_pose ;;
  pose_only) ABLATION_MODE=fixed_clock ;;
  *) echo "Experiment arm must be baseline, full, time_only, or pose_only" >&2; exit 3 ;;
esac
if [[ "$ACTION" == "teacher" || "$ACTION" == "verify" || "$ACTION" == "help" ]]; then
  EXPERIMENT_SHIFT=${ED3DGS_EXPERIMENT_SHIFT:-0}
else
  EXPERIMENT_SHIFT=${ED3DGS_EXPERIMENT_SHIFT:?ED3DGS_EXPERIMENT_SHIFT is required}
fi
case "$EXPERIMENT_SHIFT" in
  0|8|20) ;;
  *) echo "Experiment shift must be 0, 8, or 20" >&2; exit 3 ;;
esac
if [[ "$ACTION" == "teacher" || "$ACTION" == "help" || "$ACTION" == "verify" ]]; then
  SUPPORT_CONTRACT=${ED3DGS_SUPPORT_CONTRACT:-}
elif [[ -n "${ED3DGS_SUPPORT_CONTRACT:-}" ]]; then
  SUPPORT_CONTRACT=$ED3DGS_SUPPORT_CONTRACT
elif [[ "$SCENE" == "covers" ]]; then
  case "$EXPERIMENT_SHIFT" in
    0) SUPPORT_CONTRACT="$ROOT/arguments/fixed_support_shift0.json" ;;
    8) SUPPORT_CONTRACT="$ROOT/arguments/fixed_support.json" ;;
    20) SUPPORT_CONTRACT="$ROOT/arguments/fixed_support_shift20.json" ;;
  esac
else
  SCENE_SUPPORT="$ROOT/arguments/fixed_support_${SCENE}_shift${EXPERIMENT_SHIFT}.json"
  [[ -f "$SCENE_SUPPORT" ]] || {
    echo "No support contract for scene=$SCENE shift=$EXPERIMENT_SHIFT; set ED3DGS_SUPPORT_CONTRACT explicitly" >&2
    exit 4
  }
  SUPPORT_CONTRACT="$SCENE_SUPPORT"
fi
[[ "$CONFIG" = /* ]] || CONFIG="$ROOT/$CONFIG"
[[ "$TEACHER_CONFIG" = /* ]] || TEACHER_CONFIG="$ROOT/$TEACHER_CONFIG"
if [[ -n "$SUPPORT_CONTRACT" && "$SUPPORT_CONTRACT" != /* ]]; then
  SUPPORT_CONTRACT="$ROOT/$SUPPORT_CONTRACT"
fi
if [[ -n "$BASELINE_GATE" && "$BASELINE_GATE" != /* ]]; then
  BASELINE_GATE="$ROOT/$BASELINE_GATE"
fi

require_python() {
  if [[ "$PYTHON" == */* ]]; then
    [[ -x "$PYTHON" ]] || { echo "Python executable is not runnable: $PYTHON" >&2; exit 4; }
  else
    command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON" >&2; exit 4; }
  fi
}

require_gpu_seed() {
  local gpu=$1
  local seed=$2
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "GPU must be in 0..3" >&2; exit 3; }
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be an integer" >&2; exit 3; }
}

require_free_memory() {
  local gpu=$1
  local minimum=$2
  local free
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | xargs)
  (( free >= minimum )) || {
    echo "GPU $gpu has insufficient free memory: $free MiB; required: $minimum MiB" >&2
    exit 5
  }
}

strict_calibration_steps() {
  case "$1" in
    64) echo 32 ;;
    1000) echo 496 ;;
    2560) echo 1280 ;;
    30000) echo 15000 ;;
    *) echo "Unsupported strict scene-step budget: $1" >&2; return 3 ;;
  esac
}

strict_outer_iterations() {
  local scene_steps=$1
  local calibration_steps
  calibration_steps=$(strict_calibration_steps "$scene_steps")
  echo $((scene_steps + calibration_steps))
}

arm_calibration_steps() {
  if [[ "$ARM" == "baseline" ]]; then
    echo 0
  else
    strict_calibration_steps "$1"
  fi
}

arm_outer_iterations() {
  local scene_steps=$1
  local calibration_steps
  calibration_steps=$(arm_calibration_steps "$scene_steps")
  echo $((scene_steps + calibration_steps))
}

verify_code() {
  cd "$ROOT"
  sha256sum --quiet -c MANIFEST.sha256
}

configure_training() {
  local gpu=$1
  local scene_steps=$2
  local calibration_steps
  local outer_iterations
  [[ "$STRICT_SCENE_FREEZE" == "1" || "$ARM" == "baseline" ]] || {
    echo "The final release requires strict scene freeze" >&2
    return 3
  }
  calibration_steps=$(arm_calibration_steps "$scene_steps")
  outer_iterations=$(arm_outer_iterations "$scene_steps")
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  export PYTHONPATH="$ROOT/submodules/3dgs-pose:$ROOT/submodules/simple-knn:$ROOT"
  export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export ED3DGS_ITERATIONS="$outer_iterations"
  if [[ "$ARM" == "baseline" ]]; then
    unset ED3DGS_STRICT_STEP_BUDGET_V2 ED3DGS_STRICT_CALIBRATION_STEPS \
      ED3DGS_STRICT_SCENE_STEPS
  else
    export ED3DGS_STRICT_STEP_BUDGET_V2=1
    export ED3DGS_STRICT_CALIBRATION_STEPS="$calibration_steps"
    export ED3DGS_STRICT_SCENE_STEPS="$scene_steps"
  fi
  export ED3DGS_THERMAL_FRAME_SHIFT="$EXPERIMENT_SHIFT"
  export ED3DGS_SELF_CALIBRATING_CLOCK_STUDY=1
  export ED3DGS_STRICT_COMMON_SUPPORT_V33=1 ED3DGS_R25_FOURARM="$ARM"
  export ED3DGS_R25_DUAL_SUPPORT=1
  if [[ "$ARM" == "pose_only" || "$ARM" == "baseline" ]]; then
    unset ED3DGS_GEOFLOW_SOFTVOLUME_V1
  else
    export ED3DGS_GEOFLOW_SOFTVOLUME_V1=1
  fi
  if [[ "$ARM" == "baseline" ]]; then
    unset ED3DGS_BLOCK_CALIBRATION_V34
  else
    export ED3DGS_BLOCK_CALIBRATION_V34=1
  fi
  export ED3DGS_STRICT_SCENE_FREEZE_V1="$STRICT_SCENE_FREEZE"
  export ED3DGS_CAPACITY_ARM=modality_densify
  export ED3DGS_STAGE2_TEACHER_MODEL_PATH="$TEACHER"
  export ED3DGS_MODALITY_ROUTING_STABLE_V1=1
  export ED3DGS_EMA_MODAL_LOSS_V1=1 ED3DGS_CLOCK_FREEZE_AFTER_V1=15000
  export ED3DGS_MAX_GAUSSIANS=155000
  export ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT="$SUPPORT_CONTRACT"
  unset ED3DGS_MEMORY_SAFE_BACKWARD_V1 ED3DGS_MEMORY_SAFE_GAUSSIAN_THRESHOLD_V1 \
    ED3DGS_DEFORMATION_CHECKPOINT ED3DGS_R25_MATCHED_ARM ED3DGS_V33_ARM \
    ED3DGS_INTERNAL_CLOCK_V30 ED3DGS_INTERNAL_CLOCK_V34 ED3DGS_V34_RUN_TAG \
    ED3DGS_V34_AFFINE_CLOCK \
    ED3DGS_BOOTSTRAP_REPORT ED3DGS_BOOTSTRAP_SHA256 \
    ED3DGS_CLOCK_REFINEMENT_GATE_REPORT \
    ED3DGS_CLOCK_REFINEMENT_OPTIMIZATION_REPORT \
    ED3DGS_CLOCK_REFINEMENT_COARSE_OFFSET \
    ED3DGS_ZERO_START_CLOCK_PHASE_REPORT \
    ED3DGS_ZERO_START_CLOCK_PHASE_STEPS \
    ED3DGS_ZERO_START_CLOCK_PHASE ED3DGS_MOTION_CLOCK_CONTINUATION \
    ED3DGS_MATCHED_MOTION_CLOCK_STUDY ED3DGS_CLOCK_TOTAL_GRAD_V1 \
    ED3DGS_V34_JOINT_POSE_PERTURB ED3DGS_THERMAL_ENDPOINT_DRIFT_V34 \
    PYTHONOPTIMIZE
  if [[ "$ARM" == "pose_only" || "$ARM" == "baseline" ]]; then
    unset ED3DGS_JOINT_CLOCK_LOSS ED3DGS_JOINT_CLOCK_LOSS_VARIANT \
      ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1 ED3DGS_TEMPORAL_CONSENSUS_V1 \
      ED3DGS_TEMPORAL_CONSENSUS_MODE ED3DGS_SELF_CALIBRATING_CLOCK_STUDY \
      ED3DGS_CLOCK_FREEZE_AFTER_V1 ED3DGS_GEOFLOW_SOFTVOLUME_V1
  else
    export ED3DGS_JOINT_CLOCK_LOSS=1 ED3DGS_JOINT_CLOCK_LOSS_VARIANT=routed_ngf
    export ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1=1
    export ED3DGS_TEMPORAL_CONSENSUS_V1=1 ED3DGS_TEMPORAL_CONSENSUS_MODE=consensus_only
    export ED3DGS_SELF_CALIBRATING_CLOCK_STUDY=1
  fi
  if [[ -n "$ABLATION_MODE" ]]; then
    export ED3DGS_SHIFT20_ABLATION_MODE="$ABLATION_MODE"
  else
    unset ED3DGS_SHIFT20_ABLATION_MODE
  fi
  ulimit -n 65536
}

configure_evaluation() {
  local gpu=$1
  local scene_steps=${2:-0}
  local calibration_steps=0
  local outer_iterations=30000
  if [[ "$scene_steps" != "0" ]]; then
    calibration_steps=$(arm_calibration_steps "$scene_steps")
    outer_iterations=$(arm_outer_iterations "$scene_steps")
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ -n "${ED3DGS_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$ED3DGS_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  export PYTHONPATH="$ROOT/submodules/3dgs-pose:$ROOT/submodules/simple-knn:$ROOT"
  export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export ED3DGS_ITERATIONS="$outer_iterations" ED3DGS_R25_FOURARM="$ARM"
  if [[ "$ARM" == "baseline" ]]; then
    unset ED3DGS_STRICT_SCENE_FREEZE_V1
  fi
  if [[ "$scene_steps" != "0" && "$ARM" != "baseline" ]]; then
    export ED3DGS_STRICT_STEP_BUDGET_V2=1
    export ED3DGS_STRICT_CALIBRATION_STEPS="$calibration_steps"
    export ED3DGS_STRICT_SCENE_STEPS="$scene_steps"
  else
    unset ED3DGS_STRICT_STEP_BUDGET_V2 \
      ED3DGS_STRICT_CALIBRATION_STEPS ED3DGS_STRICT_SCENE_STEPS
  fi
  export ED3DGS_EVAL_THERMAL_FRAME_SHIFT="$EXPERIMENT_SHIFT"
  export ED3DGS_STRICT_COMMON_SUPPORT_V33=1
  export ED3DGS_R25_DUAL_SUPPORT=1 ED3DGS_GEOFLOW_SOFTVOLUME_V1=1
  export ED3DGS_CAPACITY_ARM=modality_densify
  export ED3DGS_STAGE2_TEACHER_MODEL_PATH="$TEACHER"
  export ED3DGS_MODALITY_ROUTING_STABLE_V1=1
  export ED3DGS_EMA_MODAL_LOSS_V1=1 ED3DGS_CLOCK_FREEZE_AFTER_V1=15000
  export ED3DGS_MAX_GAUSSIANS=155000
  export ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT="$SUPPORT_CONTRACT"
  unset ED3DGS_THERMAL_FRAME_SHIFT ED3DGS_INTERNAL_CLOCK_V30 \
    ED3DGS_INTERNAL_CLOCK_V34 ED3DGS_BLOCK_CALIBRATION_V34 \
    ED3DGS_R25_MATCHED_ARM ED3DGS_V33_ARM ED3DGS_V34_AFFINE_CLOCK \
    ED3DGS_MEMORY_SAFE_BACKWARD_V1 ED3DGS_MEMORY_SAFE_GAUSSIAN_THRESHOLD_V1 \
    ED3DGS_DEFORMATION_CHECKPOINT ED3DGS_BOOTSTRAP_REPORT \
    ED3DGS_BOOTSTRAP_SHA256 ED3DGS_V34_JOINT_POSE_PERTURB \
    ED3DGS_THERMAL_ENDPOINT_DRIFT_V34 \
    ED3DGS_MATCHED_MOTION_CLOCK_STUDY ED3DGS_CLOCK_TOTAL_GRAD_V1 \
    ED3DGS_ZERO_START_CLOCK_PHASE ED3DGS_ZERO_START_CLOCK_PHASE_STEPS \
    ED3DGS_MOTION_CLOCK_CONTINUATION \
    ED3DGS_ZERO_START_CLOCK_PHASE_REPORT \
    ED3DGS_CLOCK_REFINEMENT_GATE_REPORT \
    ED3DGS_CLOCK_REFINEMENT_OPTIMIZATION_REPORT \
    ED3DGS_CLOCK_REFINEMENT_COARSE_OFFSET PYTHONOPTIMIZE
  if [[ "$ARM" == "pose_only" || "$ARM" == "baseline" ]]; then
    unset ED3DGS_JOINT_CLOCK_LOSS ED3DGS_JOINT_CLOCK_LOSS_VARIANT \
      ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1 ED3DGS_TEMPORAL_CONSENSUS_V1 \
      ED3DGS_TEMPORAL_CONSENSUS_MODE ED3DGS_SELF_CALIBRATING_CLOCK_STUDY \
      ED3DGS_CLOCK_FREEZE_AFTER_V1 ED3DGS_GEOFLOW_SOFTVOLUME_V1
  else
    export ED3DGS_JOINT_CLOCK_LOSS=1 ED3DGS_JOINT_CLOCK_LOSS_VARIANT=routed_ngf
    export ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1=1
    export ED3DGS_SELF_CALIBRATING_CLOCK_STUDY=1
    export ED3DGS_TEMPORAL_CONSENSUS_V1=1 ED3DGS_TEMPORAL_CONSENSUS_MODE=consensus_only
  fi
  if [[ -n "$ABLATION_MODE" ]]; then
    export ED3DGS_SHIFT20_ABLATION_MODE="$ABLATION_MODE"
  else
    unset ED3DGS_SHIFT20_ABLATION_MODE
  fi
}

configure_teacher() {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  export PYTHONPATH="$ROOT/submodules/3dgs-pose:$ROOT/submodules/simple-knn:$ROOT"
  export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export ED3DGS_ITERATIONS=30000 ED3DGS_THERMAL_FRAME_SHIFT=0
  export ED3DGS_MAX_GAUSSIANS=155000
  unset ED3DGS_EVAL_THERMAL_FRAME_SHIFT ED3DGS_R25_FOURARM \
    ED3DGS_R25_MATCHED_ARM ED3DGS_V33_ARM ED3DGS_CAPACITY_ARM \
    ED3DGS_STAGE2_TEACHER_MODEL_PATH ED3DGS_JOINT_CLOCK_LOSS \
    ED3DGS_JOINT_CLOCK_LOSS_VARIANT ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1 \
    ED3DGS_TEMPORAL_CONSENSUS_V1 ED3DGS_TEMPORAL_CONSENSUS_MODE \
    ED3DGS_SELF_CALIBRATING_CLOCK_STUDY ED3DGS_STRICT_COMMON_SUPPORT_V33 \
    ED3DGS_R25_DUAL_SUPPORT ED3DGS_GEOFLOW_SOFTVOLUME_V1 \
    ED3DGS_BLOCK_CALIBRATION_V34 ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT \
    ED3DGS_STRICT_SCENE_FREEZE_V1 \
    ED3DGS_STRICT_STEP_BUDGET_V2 ED3DGS_STRICT_CALIBRATION_STEPS \
    ED3DGS_STRICT_SCENE_STEPS \
    ED3DGS_MODALITY_ROUTING_STABLE_V1 ED3DGS_EMA_MODAL_LOSS_V1 \
    ED3DGS_CLOCK_FREEZE_AFTER_V1 ED3DGS_INTERNAL_CLOCK_V30 \
    ED3DGS_INTERNAL_CLOCK_V34 ED3DGS_V34_RUN_TAG ED3DGS_V34_AFFINE_CLOCK \
    ED3DGS_MEMORY_SAFE_BACKWARD_V1 \
    ED3DGS_MEMORY_SAFE_GAUSSIAN_THRESHOLD_V1 ED3DGS_DEFORMATION_CHECKPOINT \
    ED3DGS_BOOTSTRAP_REPORT ED3DGS_BOOTSTRAP_SHA256 \
    ED3DGS_V34_JOINT_POSE_PERTURB ED3DGS_THERMAL_ENDPOINT_DRIFT_V34 \
    ED3DGS_MATCHED_MOTION_CLOCK_STUDY ED3DGS_CLOCK_TOTAL_GRAD_V1 \
    ED3DGS_ZERO_START_CLOCK_PHASE ED3DGS_ZERO_START_CLOCK_PHASE_STEPS \
    ED3DGS_MOTION_CLOCK_CONTINUATION PYTHONOPTIMIZE
  ulimit -n 65536
}

run_teacher() {
  local gpu=$1
  local seed=$2
  local log="${TEACHER}.log"
  for path in "$TEACHER" "$log"; do
    [[ ! -e "$path" ]] || { echo "Refusing existing teacher artifact: $path" >&2; exit 6; }
  done
  require_free_memory "$gpu" 22000
  mkdir -p "$(dirname "$TEACHER")" "$(dirname "$log")"
  configure_teacher "$gpu"
  cd "$ROOT"
  "$PYTHON" train.py -s "$DATASET" -m "$TEACHER" -r 2 \
    --configs "$TEACHER_CONFIG" --seed "$seed" --save_iterations 30000 \
    --checkpoint_iterations 30000 2>&1 | tee "$log"
  local model_dir="$TEACHER/point_cloud/iteration_30000"
  for path in "$model_dir/point_cloud.ply" "$model_dir/deformation.pth"; do
    [[ -s "$path" ]] || { echo "Missing RGB teacher artifact: $path" >&2; exit 4; }
  done
  printf '%s\n' \
    "scene=$SCENE" \
    'protocol=strict_rgb_only_teacher' \
    'from_scratch=true' \
    "declared_seed=$seed" \
    'iterations=30000' \
    'thermal_frame_shift=0' \
    "dataset=$DATASET" \
    "config=$TEACHER_CONFIG" > "$TEACHER/run_manifest.txt"
  sha256sum "$model_dir/point_cloud.ply" "$model_dir/deformation.pth" \
    "$TEACHER_CONFIG" "$DATASET/rgb/dataset.json" \
    > "$TEACHER/provenance.sha256"
  touch "$TEACHER/RGB_TEACHER_COMPLETE" "$TEACHER/RUN_COMPLETE"
  echo "RGB_TEACHER_COMPLETE scene=$SCENE seed=$seed output=$TEACHER"
}

lock_ground_truth() {
  local gpu=$1
  local lock_arm=full
  local lock_root="$RUN_ROOT/$SCENE/ground_truth_lock"
  local render_root="$lock_root/render"
  local render_log="$lock_root/render.log"
  local lock="$lock_root/ground_truth_lock.json"
  [[ ! -e "$lock_root" ]] || {
    echo "Refusing existing GT-lock root: $lock_root" >&2
    exit 6
  }
  for path in "$GT_LOCK_MODEL/RUN_COMPLETE" \
    "$GT_LOCK_MODEL/point_cloud/iteration_30000/point_cloud.ply" \
    "$GT_LOCK_MODEL/point_cloud/iteration_30000/deformation.pth"; do
    [[ -e "$path" ]] || { echo "Missing GT-lock reference model: $path" >&2; exit 4; }
  done
  require_free_memory "$gpu" 16000
  mkdir -p "$lock_root"
  configure_evaluation "$gpu"
  unset ED3DGS_STRICT_SCENE_FREEZE_V1
  unset ED3DGS_SHIFT20_ABLATION_MODE
  export ED3DGS_R25_FOURARM="$lock_arm"
  if [[ "$GT_LOCK_MODEL" == "$TEACHER" ]]; then
    lock_arm=baseline
    export ED3DGS_R25_FOURARM="$lock_arm"
  fi
  cd "$ROOT"
  "$PYTHON" -m time_alignment.evaluation -s "$DATASET" -m "$GT_LOCK_MODEL" -r 2 \
    --configs "$CONFIG" --iteration 30000 --output-root "$render_root" --arm "$lock_arm" \
    --support-contract "$SUPPORT_CONTRACT" \
    --expected-test-shift "$EXPERIMENT_SHIFT" 2>&1 | tee "$render_log"
  "$PYTHON" -m time_alignment.ground_truth_lock \
    --eval-root "$render_root" --render-log "$render_log" \
    --support-contract "$SUPPORT_CONTRACT" \
    --expected-shift "$EXPERIMENT_SHIFT" --output "$lock"
  echo "GROUND_TRUTH_LOCK_COMPLETE output=$lock"
}

run_gate() {
  local gpu=$1
  local seed=$2
  local output="$RUN_ROOT/$SCENE/gates/seed_${seed}"
  local log="$RUN_ROOT/$SCENE/logs/gate_seed_${seed}.log"
  [[ ! -e "$output" && ! -e "$log" ]] || {
    echo "Refusing existing Gate artifact for seed $seed" >&2
    exit 6
  }
  require_free_memory "$gpu" 22000
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  local calibration_steps
  local outer_iterations
  calibration_steps=$(arm_calibration_steps "$GATE_ITERATIONS")
  outer_iterations=$(arm_outer_iterations "$GATE_ITERATIONS")
  configure_training "$gpu" "$GATE_ITERATIONS"
  cd "$ROOT"
  printf '%s\n' 'A5000_FASTPATH_CONFIG {"deformation_checkpoint": false, "memory_safe_backward": false}' | tee "$log"
  "$PYTHON" train.py -s "$DATASET" -m "$output" -r 2 \
    --configs "$CONFIG" --seed "$seed" --save_iterations "$outer_iterations" \
    --checkpoint_iterations "$outer_iterations" 2>&1 | tee -a "$log"
  local baseline_args=()
  if [[ -n "$BASELINE_GATE" ]]; then
    baseline_args=(--baseline-gate "$BASELINE_GATE")
  fi
  local strict_args=()
  if [[ "$STRICT_SCENE_FREEZE" == "1" ]]; then
    strict_args=(--strict-scene-freeze)
  fi
  local budget_args=()
  if [[ "$ARM" != "baseline" ]]; then
    budget_args=(--calibration-steps "$calibration_steps" \
      --scene-steps "$GATE_ITERATIONS")
  fi
  "$PYTHON" -m time_alignment.gate_audit --output "$output" --log "$log" \
    --arm "$ARM" \
    --expected-shift "$EXPERIMENT_SHIFT" --expected-iterations "$outer_iterations" \
    "${budget_args[@]}" \
    --support-contract "$SUPPORT_CONTRACT" "${baseline_args[@]}" \
    "${strict_args[@]}"
  touch "$output/RUN_COMPLETE"
  echo "GATE_COMPLETE seed=$seed output=$output"
}

run_smoke() {
  local gpu=$1
  local seed=$2
  local output="$RUN_ROOT/$SCENE/mechanical/seed_${seed}"
  local log="$RUN_ROOT/$SCENE/logs/mechanical_seed_${seed}.log"
  [[ ! -e "$output" && ! -e "$log" ]] || {
    echo "Refusing existing mechanical Gate artifact for seed $seed" >&2
    exit 6
  }
  require_free_memory "$gpu" 22000
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  local calibration_steps
  local outer_iterations
  calibration_steps=$(arm_calibration_steps 64)
  outer_iterations=$(arm_outer_iterations 64)
  configure_training "$gpu" 64
  cd "$ROOT"
  printf '%s\n' 'A5000_FASTPATH_CONFIG {"deformation_checkpoint": false, "memory_safe_backward": false}' | tee "$log"
  "$PYTHON" train.py -s "$DATASET" -m "$output" -r 2 \
    --configs "$CONFIG" --seed "$seed" --save_iterations "$outer_iterations" \
    --checkpoint_iterations "$outer_iterations" 2>&1 | tee -a "$log"
  local strict_args=()
  if [[ "$STRICT_SCENE_FREEZE" == "1" ]]; then
    strict_args=(--strict-scene-freeze)
  fi
  local budget_args=()
  if [[ "$ARM" != "baseline" ]]; then
    budget_args=(--calibration-steps "$calibration_steps" --scene-steps 64)
  fi
  "$PYTHON" -m time_alignment.gate_audit --output "$output" --log "$log" \
    --arm "$ARM" \
    --expected-shift "$EXPERIMENT_SHIFT" --expected-iterations "$outer_iterations" \
    "${budget_args[@]}" \
    --support-contract "$SUPPORT_CONTRACT" --mechanical \
    "${strict_args[@]}"
  touch "$output/RUN_COMPLETE"
  echo "MECHANICAL_GATE_COMPLETE seed=$seed output=$output"
}

run_training() {
  local gpu=$1
  local seed=$2
  local gate="$RUN_ROOT/$SCENE/gates/seed_${seed}/gate_audit.json"
  "$PYTHON" - "$gate" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads(path.read_text())
if report.get("status") != "PASS" or not all(report.get("checks", {}).values()):
    raise SystemExit(f"Gate audit failed: {path}")
PY
  require_free_memory "$gpu" 22000
  local output="$RUN_ROOT/$SCENE/seed_${seed}"
  local log="$RUN_ROOT/$SCENE/logs/training_seed_${seed}.log"
  local launcher_log="$RUN_ROOT/$SCENE/logs/training_seed_${seed}.launcher.log"
  for path in "$output" "$log" "$launcher_log"; do
    [[ ! -e "$path" ]] || { echo "Refusing existing artifact: $path" >&2; exit 6; }
  done
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  local calibration_steps
  local outer_iterations
  calibration_steps=$(arm_calibration_steps 30000)
  outer_iterations=$(arm_outer_iterations 30000)
  configure_training "$gpu" 30000
  cd "$ROOT"
  printf '%s\n' 'A5000_FASTPATH_CONFIG {"deformation_checkpoint": false, "memory_safe_backward": false}' | tee "$launcher_log"
  printf '%s\n' 'A5000_FASTPATH_CONFIG {"deformation_checkpoint": false, "memory_safe_backward": false}' | tee "$log"
  "$PYTHON" train.py -s "$DATASET" -m "$output" -r 2 \
    --configs "$CONFIG" --seed "$seed" --save_iterations "$outer_iterations" \
    --checkpoint_iterations "$outer_iterations" 2>&1 | tee -a "$log"
  finalize_training "$gpu" "$seed"
}

finalize_training() {
  local gpu=$1 seed=$2
  local output="$RUN_ROOT/$SCENE/seed_${seed}"
  local log="$RUN_ROOT/$SCENE/logs/training_seed_${seed}.log"
  local launcher_log="$RUN_ROOT/$SCENE/logs/training_seed_${seed}.launcher.log"
  local gate="$RUN_ROOT/$SCENE/gates/seed_${seed}/gate_audit.json"
  local calibration_steps outer_iterations
  calibration_steps=$(arm_calibration_steps 30000)
  outer_iterations=$(arm_outer_iterations 30000)
  [[ ! -e "$output/RUN_COMPLETE" && ! -e "$output/training_audit.json" ]] || {
    echo "Refusing already finalized/audited model: $output" >&2; exit 6;
  }
  "$PYTHON" - "$gate" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get('status') != 'PASS' or not report.get('checks') or not all(report['checks'].values()):
    raise SystemExit('Gate audit failed')
PY
  configure_training "$gpu" 30000
  cd "$ROOT"
  local baseline_args=()
  if [[ -n "$BASELINE_GATE" ]]; then
    baseline_args=(--baseline-gate "$BASELINE_GATE")
  fi
  local strict_args=()
  if [[ "$STRICT_SCENE_FREEZE" == "1" ]]; then
    strict_args=(--strict-scene-freeze)
  fi
  local budget_args=()
  if [[ "$ARM" != "baseline" ]]; then
    budget_args=(--calibration-steps "$calibration_steps" --scene-steps 30000)
  fi
  "$PYTHON" -m time_alignment.training_audit \
    --output "$output" --log "$log" --launcher-log "$launcher_log" \
    --arm "$ARM" \
    --expected-shift "$EXPERIMENT_SHIFT" --expected-iterations "$outer_iterations" \
    "${budget_args[@]}" \
    --support-contract "$SUPPORT_CONTRACT" \
    "${baseline_args[@]}" "${strict_args[@]}" \
    ${ED3DGS_LEGACY_FREEZE_STATUS_RECORDS:+--legacy-freeze-status-records}
  mv "$output/chkpnt${outer_iterations}.pth" "$output/checkpoint.pth"
  mv "$output/stage2_training_result.json" "$output/training_result.json"
  mv "$output/stage2_teacher_init.json" "$output/teacher_initialization.json"
  if [[ "$ARM" == "pose_only" || "$ARM" == "baseline" ]]; then
    [[ ! -e "$output/train_change_temporal_offset.json" ]] || {
      echo "Unexpected clock state in pose_only" >&2; exit 7;
    }
  else
    mv "$output/train_change_temporal_offset.json" "$output/learned_clock.json"
  fi
  touch "$output/RUN_COMPLETE"
  echo "TRAINING_COMPLETE seed=$seed output=$output"
}

run_evaluation() {
  local gpu=$1
  local seed=$2
  local model_arg=${3:-$RUN_ROOT/$SCENE/seed_${seed}}
  local model
  if [[ "$model_arg" = /* ]]; then model="$model_arg"; else model="$ROOT/$model_arg"; fi
  local calibration_steps
  local outer_iterations
  calibration_steps=$(arm_calibration_steps 30000)
  outer_iterations=$(arm_outer_iterations 30000)
  for path in "$model/RUN_COMPLETE" "$model/training_audit.json" \
    "$model/point_cloud/iteration_${outer_iterations}/point_cloud.ply" \
    "$model/point_cloud/iteration_${outer_iterations}/deformation.pth" \
    "$model/point_cloud/iteration_${outer_iterations}/thermal_camera_state.pth" \
    "$model/checkpoint.pth"; do
    [[ -e "$path" ]] || { echo "Missing model artifact: $path" >&2; exit 4; }
  done
  "$PYTHON" - "$model/training_audit.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if report.get("status") != "PASS" or not all(report.get("checks", {}).values()):
    raise SystemExit(f"Training audit failed: {sys.argv[1]}")
PY
  require_free_memory "$gpu" 16000
  local evaluation="$RUN_ROOT/$SCENE/evaluation/seed_${seed}"
  local gate="$RUN_ROOT/$SCENE/evaluation/seed_${seed}_clock_gate"
  [[ ! -e "$evaluation" && ! -e "$gate" ]] || {
    echo "Refusing existing evaluation artifact for seed $seed" >&2
    exit 6
  }
  mkdir -p "$(dirname "$evaluation")"
  configure_evaluation "$gpu" 30000
  local model_manifest="$RUN_ROOT/$SCENE/model_seed_${seed}.sha256.pending"
  sha256sum "$model/point_cloud/iteration_${outer_iterations}/point_cloud.ply" \
    "$model/point_cloud/iteration_${outer_iterations}/deformation.pth" \
    "$model/point_cloud/iteration_${outer_iterations}/thermal_camera_state.pth" \
    "$model/checkpoint.pth" > "$model_manifest"
  cd "$ROOT"
  "$PYTHON" -m time_alignment.evaluation -s "$DATASET" -m "$model" -r 2 \
    --configs "$CONFIG" --iteration "$outer_iterations" --output-root "$gate" --arm "$ARM" \
    --support-contract "$SUPPORT_CONTRACT" \
    --expected-test-shift "$EXPERIMENT_SHIFT" --trace-gate-views 2
  [[ -f "$gate/RENDER_CLOCK_GATE_PASS" ]] || {
    echo "Renderer clock Gate did not pass" >&2
    exit 7
  }
  "$PYTHON" -m time_alignment.evaluation -s "$DATASET" -m "$model" -r 2 \
    --configs "$CONFIG" --iteration "$outer_iterations" --output-root "$evaluation" --arm "$ARM" \
    --support-contract "$SUPPORT_CONTRACT" \
    --expected-test-shift "$EXPERIMENT_SHIFT" 2>&1 | tee "$evaluation.render.log.pending"
  mv "$model_manifest" "$evaluation/model_inputs.sha256"
  mv "$evaluation.render.log.pending" "$evaluation/render.log"
  "$PYTHON" -m time_alignment.ground_truth_audit \
    --eval-root "$evaluation" --render-log "$evaluation/render.log" \
    --expected-shift "$EXPERIMENT_SHIFT" \
    --lock "$RUN_ROOT/$SCENE/ground_truth_lock/ground_truth_lock.json"
  "$PYTHON" metrics.py --model_paths "$evaluation" --rgb --thermal --batch_size 16 \
    2>&1 | tee "$evaluation/metrics.log"
  local baseline_args=()
  if [[ -n "$BASELINE_GATE" ]]; then
    baseline_args=(--baseline-gate "$BASELINE_GATE")
  fi
  local strict_args=()
  if [[ "$STRICT_SCENE_FREEZE" == "1" ]]; then
    strict_args=(--strict-scene-freeze)
  fi
  local budget_args=()
  if [[ "$ARM" != "baseline" ]]; then
    budget_args=(--calibration-steps "$calibration_steps" --scene-steps 30000)
  fi
  "$PYTHON" -m time_alignment.evaluation_audit --arm "$ARM" \
    --eval-root "$evaluation" --model-manifest "$evaluation/model_inputs.sha256" \
    --expected-shift "$EXPERIMENT_SHIFT" --expected-iteration "$outer_iterations" \
    "${budget_args[@]}" \
    --support-contract "$SUPPORT_CONTRACT" \
    "${baseline_args[@]}" "${strict_args[@]}"
  touch "$evaluation/EVAL_COMPLETE"
  echo "EVALUATION_COMPLETE seed=$seed output=$evaluation"
}

run_lifecycle() {
  local gpu=$1
  local seed=$2
  [[ -n "${ED3DGS_OUTPUT_ROOT:-}" ]] || {
    echo "Lifecycle requires an explicit ED3DGS_OUTPUT_ROOT" >&2
    exit 3
  }
  [[ ! -e "$RUN_ROOT" ]] || {
    echo "Refusing existing lifecycle output root: $RUN_ROOT" >&2
    exit 6
  }
  require_free_memory "$gpu" 22000
  lock_ground_truth "$gpu"
  run_gate "$gpu" "$seed"
  run_training "$gpu" "$seed"
  run_evaluation "$gpu" "$seed"
  test -f "$RUN_ROOT/$SCENE/gates/seed_${seed}/RUN_COMPLETE"
  test -f "$RUN_ROOT/$SCENE/seed_${seed}/RUN_COMPLETE"
  test -f "$RUN_ROOT/$SCENE/seed_${seed}/training_audit.json"
  test -f "$RUN_ROOT/$SCENE/evaluation/seed_${seed}/EVAL_COMPLETE"
  test -f "$RUN_ROOT/$SCENE/evaluation/seed_${seed}/evaluation_audit.json"
  touch "$RUN_ROOT/LIFECYCLE_COMPLETE"
  echo "SELF_CALIBRATING_RGBT_LIFECYCLE_COMPLETE arm=$ARM seed=$seed output=$RUN_ROOT"
}

case "$ACTION" in
  help|--help|-h)
    echo "usage: run.sh {teacher|smoke|gate|train|finalize|evaluate|lifecycle|verify} [GPU] [SEED] [MODEL_PATH]"
    ;;
  teacher|smoke|gate|train|finalize|evaluate|lifecycle)
    GPU=${2:?usage: run.sh $ACTION GPU SEED [MODEL_PATH]}
    SEED=${3:?usage: run.sh $ACTION GPU SEED [MODEL_PATH]}
    require_python
    require_gpu_seed "$GPU" "$SEED"
    if [[ "$ACTION" == "teacher" ]]; then
      [[ -n "$DATASET" && -d "$DATASET" && -f "$CONFIG" ]] || {
        echo "Missing dataset or scene config" >&2
        exit 4
      }
    else
      [[ -n "$DATASET" && -d "$DATASET" && -f "$CONFIG" && -f "$SUPPORT_CONTRACT" ]] || {
        echo "Missing dataset, scene config, or support contract" >&2
        exit 4
      }
    fi
    if [[ -n "$BASELINE_GATE" && ! -s "$BASELINE_GATE" ]]; then
      echo "Missing zero-shift baseline Gate: $BASELINE_GATE" >&2
      exit 4
    fi
    verify_code
    if [[ "$ACTION" == teacher ]]; then
      [[ -f "$TEACHER_CONFIG" ]] || {
        echo "Missing RGB teacher config: $TEACHER_CONFIG" >&2
        exit 4
      }
      run_teacher "$GPU" "$SEED"
      exit 0
    fi
    [[ -f "$TEACHER/RUN_COMPLETE" \
      && -f "$TEACHER/RGB_TEACHER_COMPLETE" ]] || {
      echo "Missing frozen RGB teacher: $TEACHER" >&2
      exit 4
    }
    if [[ "$ACTION" == smoke ]]; then run_smoke "$GPU" "$SEED"; fi
    if [[ "$ACTION" == gate ]]; then run_gate "$GPU" "$SEED"; fi
    if [[ "$ACTION" == train ]]; then run_training "$GPU" "$SEED"; fi
    if [[ "$ACTION" == finalize ]]; then finalize_training "$GPU" "$SEED"; fi
    if [[ "$ACTION" == evaluate ]]; then run_evaluation "$GPU" "$SEED" "${4:-}"; fi
    if [[ "$ACTION" == lifecycle ]]; then run_lifecycle "$GPU" "$SEED"; fi
    ;;
  verify)
    verify_code
    "$PYTHON" -m time_alignment.release_audit
    ;;
  *)
    echo "usage: run.sh {teacher|smoke|gate|train|finalize|evaluate|lifecycle|verify} [GPU] [SEED] [MODEL_PATH]" >&2
    exit 2
    ;;
esac
