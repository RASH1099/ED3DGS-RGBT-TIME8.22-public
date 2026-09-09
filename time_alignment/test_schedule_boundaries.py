import copy
import unittest

from time_alignment import schedule


class ScheduleBoundaryTests(unittest.TestCase):
    def test_all_budgets(self):
        for calibration, scene in [(32, 64), (496, 1000), (1280, 2560), (15000, 30000)]:
            switch = schedule.support_switch_iteration(calibration)
            phases = [schedule.phase_for_iteration(i, calibration_steps=calibration,
                      scene_steps=scene) for i in range(1, calibration + scene + 1)]
            self.assertEqual(phases.count("calibration"), calibration)
            self.assertEqual(phases.count("reconstruction") + phases.count("scene_tail"), scene)
            self.assertEqual(phases[switch - 1], "scene_tail")
            self.assertEqual(phases[switch - 2], "reconstruction")

    def test_formal_freeze_and_switch(self):
        self.assertEqual(schedule.support_switch_iteration(15000), 30001)
        kwargs = dict(calibration_steps=15000, scene_steps=30000)
        self.assertEqual(schedule.phase_for_iteration(29992, **kwargs), "calibration")
        self.assertEqual(schedule.phase_for_iteration(29993, **kwargs), "reconstruction")
        self.assertEqual(schedule.phase_for_iteration(45000, **kwargs), "scene_tail")

    def test_freeze_records(self):
        row = dict(iteration=29992, clock_optimizer_steps=15000,
                   offset_frames=19.986, requires_grad_next=False)
        self.assertTrue(schedule.freeze_records_valid([row],29992,45000,15000,19.986))
        legacy = [dict(row, iteration=i) for i in range(29992, 45001)]
        self.assertTrue(schedule.freeze_records_valid(legacy,29992,45000,15000,19.986,True))
        self.assertFalse(schedule.freeze_records_valid(legacy,29992,45000,15000,19.986))
        for key, value in [("clock_optimizer_steps",15001),("offset_frames",20.1),
                           ("requires_grad_next",True),("iteration",29991)]:
            bad = copy.deepcopy(legacy)
            bad[100][key] = value
            self.assertFalse(schedule.freeze_records_valid(bad,29992,45000,15000,19.986,True))
        self.assertFalse(schedule.freeze_records_valid(legacy[:-1],29992,45000,15000,19.986,True))


if __name__ == "__main__":
    unittest.main()
