"""CPU-only sequencing checks; these do not validate NPU/tool compatibility."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import importlib.util
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location("urma_hw_profile",Path(__file__).resolve().parents[1]/"scripts/urma_hw_profile.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
capture_forward, MstxRange = module.capture_forward, module.MstxRange


class CaptureTests(unittest.TestCase):
    def fixtures(self):
        events = []

        class Guard:
            def begin(self):
                events.append("guard.begin")

            def before_forward(self):
                events.append("align")

            def end(self):
                events.append("guard.end")

        class Marker:
            @contextmanager
            def capture(self):
                events.append("range.begin")
                try:
                    yield
                finally:
                    events.append("range.end")

        return events, Guard(), Marker()

    def test_only_last_forward_is_selected(self):
        events, guard, marker = self.fixtures()
        capture_forward(lambda: events.append("forward"), guard, marker, 10)
        self.assertEqual(events, ["guard.begin"] + ["align", "forward"] * 10 + ["align"] +
                         ["range.begin", "forward", "range.end", "guard.end"])

    def test_zero_warmup_still_captures_one(self):
        events, guard, marker = self.fixtures()
        capture_forward(lambda: events.append("forward"), guard, marker, 0)
        self.assertEqual(events, ["guard.begin", "align", "range.begin", "forward",
                                  "range.end", "guard.end"])

    def test_failure_closes_range_without_entering_tail_collective(self):
        events, guard, marker = self.fixtures()

        def fail():
            raise RuntimeError("launch failed")

        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            capture_forward(fail, guard, marker, 0)
        self.assertEqual(events, ["guard.begin", "align", "range.begin", "range.end"])

    def test_mstx_zero_id_is_closed(self):
        with patch.object(module.ctypes,"CDLL") as load:
            load.return_value.mstxRangeStartA.return_value = 0
            marker = MstxRange()
            with marker.capture():
                pass
            load.return_value.mstxRangeStartA.assert_called_once_with(b"urma_capture", None)
            load.return_value.mstxRangeEnd.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
