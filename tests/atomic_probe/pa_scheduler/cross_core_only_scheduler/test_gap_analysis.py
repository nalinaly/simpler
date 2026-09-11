import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gaps", Path(__file__).with_name("analyze_swimlane_gaps.py"))
gaps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gaps)


class GapAnalysisTest(unittest.TestCase):
    def test_nested_not_double_counted(self):
        result = gaps.partition(0, 10, [(1, 9, "ExecBind"), (2, 4, "Atomic"), (5, 6, "Dcci")])
        self.assertEqual(result["Unattributed"], 2)
        self.assertEqual(result["ExecBind"], 5)
        self.assertEqual(result["Atomic"], 2)
        self.assertEqual(result["Dcci"], 1)
        self.assertEqual(sum(result.values()), 10)

    def test_clip_to_scheduler_window(self):
        result = gaps.partition(3, 7, [(0, 5, "ExecFanin"), (5, 9, "ExecComplete")])
        self.assertEqual(result["ExecFanin"], 2)
        self.assertEqual(result["ExecComplete"], 2)
        self.assertEqual(result["Unattributed"], 0)

    def test_adjacent_atomic_intervals(self):
        result = gaps.partition(0, 4, [(0, 1, "Atomic"), (1, 4, "Atomic")])
        self.assertEqual(result["Atomic"], 4)
        self.assertEqual(result["Unattributed"], 0)

    def test_empty_is_unattributed_not_idle(self):
        result = gaps.partition(0, 5, [])
        self.assertEqual(result["Unattributed"], 5)
