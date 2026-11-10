"""CPU-only sequencing checks; not NPU/tool compatibility validation."""
import importlib.util
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("urma_hw_profile", SCRIPTS / "urma_hw_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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

        return events, Guard()

    def test_exactly_one_forward_with_alignment(self):
        events, guard = self.fixtures()
        module.capture_forward(lambda: events.append("forward"), guard)
        self.assertEqual(events, ["guard.begin", "align", "forward", "guard.end"])

    def test_failure_does_not_enter_tail_collective(self):
        events, guard = self.fixtures()

        def fail():
            raise RuntimeError("launch failed")

        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            module.capture_forward(fail, guard)
        self.assertEqual(events, ["guard.begin", "align"])

    def test_launcher_uses_kernel_filter_without_mstx_or_warmup(self):
        script = (SCRIPTS / "run_urma_hw_profile.sh").read_text()
        self.assertIn("--mstx=false", script)
        self.assertNotIn("--mstx-include", script)
        self.assertNotIn("URMA_HW_WARMUP", script)
        self.assertIn("--warm-up=0", script)
        self.assertIn("--launch-count=1", script)
        self.assertIn("--kernel-name='*mega_moe_urma_kernel*'", script)


if __name__ == "__main__":
    unittest.main()
