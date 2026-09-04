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
    'WhiteFoamCovers': 536,
}

SCENE = os.environ.get('ED3DGS_SCENE', '')
if SCENE not in FRAME_COUNTS:
    raise ValueError(f'Unsupported MeetingRoom scene: {SCENE!r}')
NUM_FRAMES = FRAME_COUNTS[SCENE]
MIN_EMBEDDINGS = (NUM_FRAMES + 9) // 10

ITERATIONS = int(os.environ.get('ED3DGS_ITERATIONS', '30000'))
if ITERATIONS != 30_000:
    raise ValueError('A formal MeetingRoom RGB teacher requires 30000 iterations')

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
    rgb_only_teacher=True,
    thermal_only=False,
    no_thermal_pose_opt=True,
    thermal_pose_lr_r=0.0,
    thermal_pose_lr_t=0.0,
    thermal_intrinsic_lr=0.0,
    thermal_intrinsic_prior_weight=0.0,
    thermal_loss_weight=0.0,
    thermal_feature_lr=0.0,
    thermal_opacity_lr=0.0,
    modality_lr=0.0,
    enable_modality_densify=False,
    temporal_alignment_enabled=False,
    temporal_offset_lr=0.0,
    densify_from_iter=500,
    densify_until_iter=30_000,
    densification_interval=100,
    pruning_from_iter=500,
    pruning_interval=100,
    position_lr_max_steps=30_000,
    deformation_lr_max_steps=30_000,
    c2w_lr=0,
    c2w_lr_final=0,
)
