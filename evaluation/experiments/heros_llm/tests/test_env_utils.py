"""Unit tests for local .env loading helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evaluation.experiments.heros_llm.env_utils import parse_env_file


class EnvUtilsTests(unittest.TestCase):
    def test_parse_env_file_reads_key_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "# comment\nOPENAI_API_KEY=test-key\nexport MODEL_NAME='gpt-4.1-mini'\n",
                encoding="utf-8",
            )
            parsed = parse_env_file(env_path)
            self.assertEqual(parsed["OPENAI_API_KEY"], "test-key")
            self.assertEqual(parsed["MODEL_NAME"], "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
