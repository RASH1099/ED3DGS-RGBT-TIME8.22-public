import unittest

from time_alignment import ablation_audit, gate_audit, training_audit


class AuditLogParsingTest(unittest.TestCase):
    def test_rows_accept_tqdm_carriage_return_boundaries(self):
        log = (
            'CLOCK_STEP_NUISANCE_AUDIT {"iteration": 1}\n'
            '\rTraining progress: 50%| 15000/30000 '
            '[2:19:24<2:51:12, pts=147753]'
            'CLOCK_STEP_NUISANCE_AUDIT {"iteration": 15000}\r\n'
        )
        for module in (training_audit, gate_audit, ablation_audit):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    [row["iteration"] for row in module.rows(
                        log, "CLOCK_STEP_NUISANCE_AUDIT")],
                    [1, 15000],
                )

    def test_rows_reject_embedded_or_wrong_event_names(self):
        log = (
            'prefix CLOCK_STEP_NUISANCE_AUDIT {"iteration": 1}\n'
            'CLOCK_STEP_NUISANCE_AUDIT_WRONG {"iteration": 15000}\n'
        )
        for module in (training_audit, gate_audit, ablation_audit):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module.rows(log, "CLOCK_STEP_NUISANCE_AUDIT"), [])


if __name__ == "__main__":
    unittest.main()
