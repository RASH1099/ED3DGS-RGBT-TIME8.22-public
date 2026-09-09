"""Validate the registered shift-zero reference used for native clock residuals."""
import hashlib
import math
import re
import json
from pathlib import Path

from time_alignment import schedule


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_matched_clock_reference(
        report, expected_support_sha256, teacher_deformation_sha256):
    finite = lambda value: isinstance(value, (int, float)) and math.isfinite(value)
    shift0 = report.get('shift0_reference', {})
    shift20 = report.get('shift20_replay', {})
    optimizer = report.get('clock_optimizer', {})
    crosschecks = report.get('archived_run_crosschecks', {})
    profiles = report.get('profile_best_offsets_frames', {})
    endpoints = report.get('final_endpoint_offsets_frames', [])
    teacher = report.get('teacher_deformation_sha256')
    code = report.get('production_code', {})
    root = Path(__file__).resolve().parents[1]
    checks = [
        report.get('status') == 'PASS',
        report.get('scene') == 'DeliverIcePacks',
        report.get('seed') == 6666,
        report.get('arm') == 'time_only',
        report.get('expected_shift_frames') == 0,
        report.get('support_contract_sha256') == expected_support_sha256,
        isinstance(teacher, str) and re.fullmatch('[0-9a-f]{64}', teacher) is not None,
        teacher_deformation_sha256 is None or teacher == teacher_deformation_sha256,
        optimizer == {
            'parameterization': '24*tanh(raw)',
            'initial_raw': 0.0,
            'optimizer': 'torch.optim.Adam',
            'learning_rate': 0.005,
            'epsilon': 1.0e-15,
            'clock_loss_weight': 0.05,
            'ramp_steps': 200,
            'matched_clock_steps': 15000,
        },
        shift0.get('requested_shift_frames') == 0,
        shift20.get('requested_shift_frames') == 20,
        shift0.get('shifted_camera_count') == 0,
        shift20.get('shifted_camera_count') == 660,
        finite(report.get('final_offset_frames')),
        report.get('final_offset_frames') == shift0.get('offset_after_15000_steps'),
        len(endpoints) == 2 and all(
            value == report.get('final_offset_frames') for value in endpoints),
        profiles == shift0.get('profile_best_offsets'),
        set(profiles) == {'full', 'left', 'right', 'early', 'late'},
        all(finite(value) for value in profiles.values()),
        report.get('matched_relative_recovery_frames')
        == shift20.get('offset_after_15000_steps')
        - shift0.get('offset_after_15000_steps'),
        report.get('target_relative_recovery_frames') == 20.0,
        finite(report.get('absolute_error_frames')),
        report.get('absolute_error_frames') <= 0.25,
        report.get('clock_accuracy_pass') is True,
        set(crosschecks) == {'shift0_496', 'shift20_496', 'shift20_15000'},
        all(value.get('pass') is True
            and finite(value.get('absolute_error_frames'))
            and value.get('absolute_error_frames') <= 1.0e-6
            for value in crosschecks.values()),
        code.get('motion_cost_volume_sha256')
        == _sha256(root / 'time_alignment' / 'motion_cost_volume.py'),
        code.get('train_sha256') == _sha256(root / 'train.py'),
    ]
    if not all(checks):
        raise ValueError('Invalid matched-budget zero-shift reference')
    return True


def validate_reference(report, support_sha256, teacher_deformation_sha256=None):
    if not isinstance(report, dict):
        raise ValueError('Missing zero-shift reference')
    protocol_path = Path(__file__).resolve().parents[1] / 'arguments' / 'reference_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    expected_support_sha256 = protocol.get(support_sha256, support_sha256)
    if report.get('schema') == 'delivericepacks_matched_clock_reference_v1':
        return _validate_matched_clock_reference(
            report, expected_support_sha256, teacher_deformation_sha256)
    checks = report.get('checks', {})
    ignored = {'clock_accuracy', 'profile_consistency'}
    required = {'arm_contract', 'shift_protocol', 'teacher_immutable', 'raw_zero',
                'global_clock', 'gradient_isolation', 'clock_observable',
                'clock_optimizer', 'clock_capture', 'strict_scene_freeze',
                'support_transition_preflight', 'iterations', 'optimizer_counts',
                'deformation_trainable', 'clean_log'}
    teacher = report.get('teacher_deformation_sha256')
    profiles = report.get('profile_best_offsets_frames', {})
    endpoints = report.get('final_endpoint_offsets_frames', [])
    finite = lambda value: isinstance(value, (int, float)) and math.isfinite(value)
    if (report.get('schema') != 'covers_self_calibrating_global_clock_gate'
            or report.get('arm') != 'time_only'
            or report.get('mechanical') is not False
            or report.get('expected_shift_frames') != 0
            or report.get('strict_scene_freeze') is not True
            or report.get('strict_step_budget_v2') is not True
            or report.get('iterations') != 1496
            or report.get('expected_calibration_steps') != 496
            or report.get('expected_scene_steps') != 1000
            or report.get('support_contract_sha256') != expected_support_sha256
            or not isinstance(teacher, str) or re.fullmatch('[0-9a-f]{64}',teacher) is None
            or (teacher_deformation_sha256 is not None and teacher != teacher_deformation_sha256)
            or not finite(report.get('final_offset_frames'))
            or len(endpoints) != 2 or not all(finite(v) for v in endpoints)
            or set(profiles) != {'full','left','right','early','late'}
            or not all(finite(v) for v in profiles.values())
            or not required.issubset(checks)
            or not ignored.issubset(checks)
            or not all(v is True for k,v in checks.items() if k not in ignored)
            or report.get('status') != ('PASS' if all(checks.values()) else 'FAIL')):
        raise ValueError('Invalid registered zero-shift reference')
    schedule.strict_block_step_counts(1496,calibration_steps=496,scene_steps=1000)
    return True


def reference_valid(report, support_sha256, teacher_deformation_sha256=None):
    try:
        return validate_reference(report,support_sha256,teacher_deformation_sha256)
    except ValueError:
        return False
