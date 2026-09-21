import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_table6_vlm_tta.sh"


class TestTable6VLMRunner(unittest.TestCase):
    def _run_parse_only(self, *args):
        env = os.environ.copy()
        env["TABLE6_PARSE_ONLY"] = "1"
        env["CLIP_CACHE_ROOT"] = str(PROJECT_ROOT / "missing-clip-cache-for-test")
        return subprocess.run(
            ["bash", str(RUNNER), *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )

    def test_method_and_mode_scope_to_single_atlas_zero_shot_row(self):
        result = self._run_parse_only("--method", "atlas", "--mode", "zs")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Prompt Settings: zs", result.stdout)
        self.assertIn("Methods:         atlas", result.stdout)
        self.assertIn("Publish Tables:  0", result.stdout)

    def test_mode_sz_is_zero_shot_alias(self):
        result = self._run_parse_only("--method", "atlas", "--mode", "sz")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Prompt Settings: zs", result.stdout)
        self.assertIn("Methods:         atlas", result.stdout)

    def test_method_and_mode_scope_to_single_atlas_coop_row(self):
        result = self._run_parse_only("--method", "atlas", "--mode", "coop")

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Prompt Settings: coop", result.stdout)
        self.assertIn("Methods:         atlas", result.stdout)
        self.assertIn("Publish Tables:  0", result.stdout)

    def test_invalid_mode_fails_before_any_experiment(self):
        result = self._run_parse_only("--method", "atlas", "--mode", "bad")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Unsupported prompt setting", result.stdout)

    def test_atlas_online_runtime_is_not_a_live_table6_path(self):
        env = os.environ.copy()
        env["TABLE6_PARSE_ONLY"] = "1"
        env["ATLAS_VLM_RUNTIME"] = "online"
        result = subprocess.run(
            ["bash", str(RUNNER), "--method", "atlas", "--mode", "zs"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Unsupported ATLAS_VLM_RUNTIME", result.stdout)

    def test_json_source_signature_is_passed_without_literal_interpolation(self):
        runner_text = RUNNER.read_text(encoding="utf-8")

        self.assertNotIn("import hashlib", runner_text)
        self.assertIn('source_signature = os.environ["SOURCE_SIGNATURE"]', runner_text)
        self.assertIn('"source_signature": source_signature', runner_text)


if __name__ == "__main__":
    unittest.main()
