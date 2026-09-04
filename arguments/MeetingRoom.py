import json
import os


_base_ = './default.py'

FRAME_COUNTS = {
    'Heatingtable': 345,
    'HotPressMachine': 624,
    'Hotwind': 768,
    'Bacon': 608,
    'DeliverIcePacks': 680,
    'HairDryer': 560,
    'HairDryerDark': 736,
    'IroningClothes': 752,
    'LightTheCandles': 576,
    'PourHotWater': 808,
    'WhiteFoamCovers': 536,
}

SCENE = os.environ.get('ED3DGS_SCENE', '')
if SCENE not in FRAME_COUNTS:
    raise ValueError(f'Unsupported MeetingRoom scene: {SCENE!r}')
NUM_FRAMES = FRAME_COUNTS[SCENE]
MIN_EMBEDDINGS = (NUM_FRAMES + 9) // 10

ARM = os.environ.get('ED3DGS_R25_FOURARM', '')
if ARM not in {'baseline', 'time_only', 'pose_only', 'full'}:
    raise ValueError(f'Invalid ED3DGS_R25_FOURARM: {ARM!r}')
ITERATIONS = int(os.environ.get('ED3DGS_ITERATIONS', '30000'))
STRICT_STEP_BUDGET_V2 = (
    os.environ.get('ED3DGS_STRICT_STEP_BUDGET_V2', '0') == '1')
if STRICT_STEP_BUDGET_V2:
    if ITERATIONS not in {96, 1496, 3840, 45000}:
        raise ValueError(
            'Strict step-budget v2 requires 96, 1496, 3840 or 45000 outer iterations')
else:
    if ITERATIONS not in {64, 1000, 2560, 30000}:
        raise ValueError('ED3DGS_ITERATIONS must be 64, 1000, 2560 or 30000')
CAPACITY_ARM = os.environ.get('ED3DGS_CAPACITY_ARM', '')
if CAPACITY_ARM not in {'shared_densify', 'modality_densify'}:
    raise ValueError(f'Invalid ED3DGS_CAPACITY_ARM: {CAPACITY_ARM!r}')

JOINT_CLOCK = os.environ.get('ED3DGS_JOINT_CLOCK_LOSS', '0') == '1'
TEMPORAL_CONSENSUS = (
    os.environ.get('ED3DGS_TEMPORAL_CONSENSUS_V1', '0') == '1')
SELF_CALIBRATING_CLOCK = (
    os.environ.get('ED3DGS_SELF_CALIBRATING_CLOCK_STUDY', '0') == '1')
CONSENSUS_MODE = os.environ.get(
    'ED3DGS_TEMPORAL_CONSENSUS_MODE', 'consensus_only')
LOSS_VARIANT = os.environ.get(
    'ED3DGS_JOINT_CLOCK_LOSS_VARIANT', 'combined')
CLOCK_FREEZE_AFTER = int(os.environ.get(
    'ED3DGS_CLOCK_FREEZE_AFTER_V1', '15000'))

if TEMPORAL_CONSENSUS and CONSENSUS_MODE not in {
        'consensus_only', 'consensus_ngf'}:
    raise ValueError(f'Invalid temporal consensus mode: {CONSENSUS_MODE!r}')
if JOINT_CLOCK and LOSS_VARIANT not in {'mind', 'ngf', 'routed_ngf'}:
    raise ValueError(f'Invalid joint clock loss variant: {LOSS_VARIANT!r}')
if SELF_CALIBRATING_CLOCK and not (JOINT_CLOCK and TEMPORAL_CONSENSUS):
    raise ValueError(
        'Self-calibrating clock requires joint clock and temporal consensus')

TIME_ENABLED = ARM in {'time_only', 'full'}
POSE_ENABLED = ARM in {'pose_only', 'full'}
CALIBRATION_ENABLED = TIME_ENABLED or POSE_ENABLED
MODALITY_DENSIFY = CAPACITY_ARM == 'modality_densify'
STRICT_SCENE_FREEZE = (
    os.environ.get('ED3DGS_STRICT_SCENE_FREEZE_V1', '0') == '1')
if STRICT_SCENE_FREEZE and not CALIBRATION_ENABLED:
    raise ValueError('Strict scene freeze requires a calibration arm')
_STRICT_BUDGET_DEFAULTS = {
    96: (32, 64),
    1496: (496, 1000),
    3840: (1280, 2560),
    45000: (15000, 30000),
}
if STRICT_STEP_BUDGET_V2:
    _default_calibration, _default_scene = _STRICT_BUDGET_DEFAULTS[ITERATIONS]
    STRICT_CALIBRATION_STEPS = int(os.environ.get(
        'ED3DGS_STRICT_CALIBRATION_STEPS', _default_calibration))
    STRICT_SCENE_STEPS = int(os.environ.get(
        'ED3DGS_STRICT_SCENE_STEPS', _default_scene))
    if (ITERATIONS != STRICT_CALIBRATION_STEPS + STRICT_SCENE_STEPS
            or STRICT_CALIBRATION_STEPS % 8 != 0
            or STRICT_SCENE_STEPS < STRICT_CALIBRATION_STEPS):
        raise ValueError('Invalid strict step-budget v2 contract')
else:
    STRICT_CALIBRATION_STEPS = sum(
        1 for iteration in range(1, ITERATIONS)
        if ((iteration - 1) // 8) % 2 == 0)
    STRICT_SCENE_STEPS = sum(
        1 for iteration in range(1, ITERATIONS)
        if ((iteration - 1) // 8) % 2 == 1)
TEACHER_PATH = os.environ.get('ED3DGS_STAGE2_TEACHER_MODEL_PATH', '')
if not TEACHER_PATH:
    raise ValueError('ED3DGS_STAGE2_TEACHER_MODEL_PATH is required')

ModelParams = dict(
    stage2_teacher_model_path=TEACHER_PATH,
    stage2_teacher_iteration=30_000,
    stage2_temporal_bootstrap_report='',
    stage2_temporal_bootstrap_sha256='',
)

ModelHiddenParams = dict(
    min_embeddings=MIN_EMBEDDINGS,
    max_embeddings=5 * MIN_EMBEDDINGS,
    c2f_temporal_iter=20_000,
    total_num_frames=NUM_FRAMES,
    change_thermal_geo=False,
)

OptimizationParams = dict(
    maxtime=NUM_FRAMES,
    iterations=ITERATIONS,
    strict_step_budget_v2=STRICT_STEP_BUDGET_V2,
    strict_calibration_steps=STRICT_CALIBRATION_STEPS,
    strict_scene_steps=STRICT_SCENE_STEPS,
    rgb_only_teacher=False,
    thermal_only=False,
    no_thermal_pose_opt=not POSE_ENABLED,
    thermal_pose_lr_r=1e-4 if POSE_ENABLED else 0.0,
    thermal_pose_lr_t=1e-4 if POSE_ENABLED else 0.0,
    thermal_pose_gray_grad_only=POSE_ENABLED,
    thermal_pose_shared_by_side=POSE_ENABLED,
    thermal_pose_start_iter=1,
    thermal_pose_freeze_after=-1,
    thermal_pose_target_steps=(
        STRICT_CALIBRATION_STEPS if STRICT_STEP_BUDGET_V2 else -1),
    thermal_intrinsic_lr=0.0,
    thermal_intrinsic_start_iter=6_000,
    thermal_intrinsic_freeze_after=10_000,
    thermal_intrinsic_shared_by_side=True,
    thermal_intrinsic_tied_aspect=False,
    thermal_intrinsic_max_rel_change=0.08,
    thermal_intrinsic_prior_weight=0.0,
    calibration_reconstruction_alternation=CALIBRATION_ENABLED,
    calibration_reconstruction_phase_length=8,
    scene_optimizer_every_iteration=not STRICT_SCENE_FREEZE,
    scene_optimizer_target_steps=(
        STRICT_SCENE_STEPS if STRICT_SCENE_FREEZE else ITERATIONS),
    enable_modality_densify=MODALITY_DENSIFY,
    densify_from_iter=500,
    densify_until_iter=30_000,
    densification_interval=100,
    pruning_from_iter=500,
    pruning_interval=100,
    position_lr_max_steps=30_000,
    deformation_lr_max_steps=30_000,
    modality_stage_a_until=(1_000 if MODALITY_DENSIFY else 1_000_000),
    modality_stage_b_until=(10_000 if MODALITY_DENSIFY else 1_000_000),
    modality_stage_c_until=(20_000 if MODALITY_DENSIFY else 1_000_000),
    c2w_lr=0,
    c2w_lr_final=0,
    temporal_alignment_enabled=TIME_ENABLED,
    temporal_offset_lr=(
        0.005 if TEMPORAL_CONSENSUS else 0.001 if JOINT_CLOCK else 0.005),
    temporal_offset_max_frames=24.0,
    temporal_offset_start_iter=1,
    temporal_offset_freeze_after=(
        CLOCK_FREEZE_AFTER if JOINT_CLOCK and not STRICT_STEP_BUDGET_V2 else -1),
    temporal_offset_target_steps=(
        STRICT_CALIBRATION_STEPS if STRICT_STEP_BUDGET_V2 else -1),
    temporal_offset_force_frames=None,
    temporal_offset_prior_weight=0.0,
    temporal_strict_common_support=True,
    temporal_affine_clock_enabled=False,
    temporal_drift_max_endpoint_frames=4.0,
    joint_clock_loss_enabled=JOINT_CLOCK,
    joint_clock_loss_weight=0.05,
    joint_clock_loss_start_iter=(1 if TEMPORAL_CONSENSUS else 200),
    joint_clock_loss_ramp_iters=(200 if TEMPORAL_CONSENSUS else 800),
    joint_clock_loss_interval=1,
)

print('FORMAL_MEETINGROOM_CONFIG ' + json.dumps({
    'scene': SCENE,
    'num_frames': NUM_FRAMES,
    'arm': ARM,
    'iterations': ITERATIONS,
    'capacity_arm': CAPACITY_ARM,
    'joint_clock': JOINT_CLOCK,
    'loss_variant': LOSS_VARIANT if JOINT_CLOCK else None,
    'temporal_consensus': TEMPORAL_CONSENSUS,
    'self_calibrating_clock': SELF_CALIBRATING_CLOCK,
    'consensus_mode': CONSENSUS_MODE if TEMPORAL_CONSENSUS else None,
    'clock_freeze_after': CLOCK_FREEZE_AFTER if JOINT_CLOCK else -1,
    'strict_scene_freeze': STRICT_SCENE_FREEZE,
    'strict_step_budget_v2': STRICT_STEP_BUDGET_V2,
    'strict_calibration_steps': STRICT_CALIBRATION_STEPS,
    'strict_scene_steps': STRICT_SCENE_STEPS,
    'scene_optimizer_target_steps': (
        STRICT_SCENE_STEPS if STRICT_SCENE_FREEZE else ITERATIONS),
    'teacher_path': TEACHER_PATH,
}, sort_keys=True))
