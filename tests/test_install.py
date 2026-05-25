import json
import tempfile
import unittest
from pathlib import Path

from scripts.install import deepseek_model_metadata, write_model_catalog


class InstallTests(unittest.TestCase):
    def test_metadata_does_not_claim_context_window(self):
        metadata = deepseek_model_metadata("deepseek-v4-pro")

        self.assertNotIn("context_window", metadata)
        self.assertNotIn("max_context_window", metadata)
        self.assertNotIn("effective_context_window_percent", metadata)

    def test_metadata_does_not_force_backend_identity(self):
        metadata = deepseek_model_metadata("deepseek-v4-pro")
        instructions = metadata["model_messages"]["instructions_template"]

        self.assertNotIn("DeepSeek V4 Pro configured", instructions)
        self.assertNotIn("deepseek-v4-pro through", instructions)

    def test_catalog_contains_only_custom_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = Path(temp_dir)
            (codex_home / "models_cache.json").write_text(
                json.dumps({"models": [{"slug": "gpt-5.5"}]}),
                encoding="utf-8",
            )

            catalog_path = write_model_catalog(codex_home, "deepseek-v4-pro")
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [model["slug"] for model in catalog["models"]],
            ["deepseek-v4-pro"],
        )


if __name__ == "__main__":
    unittest.main()
