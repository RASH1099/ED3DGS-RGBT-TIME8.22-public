import copy
import unittest
from pathlib import Path
import time_alignment.reference_contract as reference_contract
from time_alignment.reference_contract import validate_reference


class ReferenceTests(unittest.TestCase):
    def report(self):
        names = ['arm_contract','shift_protocol','teacher_immutable','raw_zero',
                 'global_clock','gradient_isolation','clock_observable','clock_optimizer',
                 'clock_capture','strict_scene_freeze','support_transition_preflight',
                 'iterations','optimizer_counts','deformation_trainable','clean_log']
        checks = dict.fromkeys(names,True)
        checks.update(clock_accuracy=False,profile_consistency=False)
        return dict(schema='covers_self_calibrating_global_clock_gate',arm='time_only',
            mechanical=False,expected_shift_frames=0,strict_scene_freeze=True,
            strict_step_budget_v2=True,iterations=1496,expected_calibration_steps=496,
            expected_scene_steps=1000,support_contract_sha256='a'*64,
            teacher_deformation_sha256='b'*64,final_offset_frames=-1.98,
            final_endpoint_offsets_frames=[-1.98,-1.98],
            profile_best_offsets_frames=dict.fromkeys(['full','left','right','early','late'],-2),
            status='FAIL',checks=checks)

    def test_native_residual_preserves_fail_status(self):
        self.assertTrue(validate_reference(self.report(),'a'*64,'b'*64))

    def test_reject_invalid_reference(self):
        for key,value in [('expected_scene_steps',30000),('iterations',45000),
                          ('mechanical',True),('strict_step_budget_v2',False),
                          ('support_contract_sha256','c'*64),('teacher_deformation_sha256','c'*64),
                          ('status','PASS'),('expected_shift_frames',20)]:
            report=self.report()
            report[key]=value
            with self.assertRaises(ValueError):
                validate_reference(report,'a'*64,'b'*64)
        report=self.report()
        report['checks']['gradient_isolation']=False
        with self.assertRaises(ValueError):
            validate_reference(report,'a'*64,'b'*64)

    def matched_report(self):
        root = Path(reference_contract.__file__).resolve().parents[1]
        shift0 = {
            'requested_shift_frames': 0,
            'shifted_camera_count': 0,
            'offset_after_15000_steps': -1.85,
            'profile_best_offsets': {
                'full': -2, 'left': -2, 'right': 0,
                'early': -2, 'late': 0,
            },
        }
        shift20 = {
            'requested_shift_frames': 20,
            'shifted_camera_count': 660,
            'offset_after_15000_steps': 18.015,
        }
        crosschecks = {
            name: {'pass': True, 'absolute_error_frames': 0.0}
            for name in ('shift0_496', 'shift20_496', 'shift20_15000')
        }
        return {
            'schema': 'delivericepacks_matched_clock_reference_v1',
            'status': 'PASS', 'scene': 'DeliverIcePacks', 'seed': 6666,
            'arm': 'time_only', 'expected_shift_frames': 0,
            'support_contract_sha256': 'a' * 64,
            'teacher_deformation_sha256': 'b' * 64,
            'clock_optimizer': {
                'parameterization': '24*tanh(raw)', 'initial_raw': 0.0,
                'optimizer': 'torch.optim.Adam', 'learning_rate': 0.005,
                'epsilon': 1.0e-15, 'clock_loss_weight': 0.05,
                'ramp_steps': 200, 'matched_clock_steps': 15000,
            },
            'production_code': {
                'motion_cost_volume_sha256': reference_contract._sha256(
                    root / 'time_alignment' / 'motion_cost_volume.py'),
                'train_sha256': reference_contract._sha256(root / 'train.py'),
            },
            'shift0_reference': shift0, 'shift20_replay': shift20,
            'final_offset_frames': -1.85,
            'final_endpoint_offsets_frames': [-1.85, -1.85],
            'profile_best_offsets_frames': shift0['profile_best_offsets'],
            'matched_relative_recovery_frames': (
                shift20['offset_after_15000_steps']
                - shift0['offset_after_15000_steps']),
            'target_relative_recovery_frames': 20.0,
            'absolute_error_frames': 0.135,
            'clock_accuracy_pass': True,
            'archived_run_crosschecks': crosschecks,
        }

    def test_accept_matched_clock_reference(self):
        self.assertTrue(validate_reference(
            self.matched_report(), 'a' * 64, 'b' * 64))

    def test_reject_tampered_matched_clock_reference(self):
        report = self.matched_report()
        report['shift0_reference']['offset_after_15000_steps'] = -1.7
        with self.assertRaises(ValueError):
            validate_reference(report, 'a' * 64, 'b' * 64)


if __name__=='__main__':
    unittest.main()
