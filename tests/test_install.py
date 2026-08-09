import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.install import (
    deepseek_model_metadata,
    patch_codex_config,
    write_launch_agent,
    write_model_catalog,
)


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

    def test_installer_writes_current_profile_file_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            codex_home = Path(temp_dir)
            catalog_path = codex_home / "deepseek_model_catalog.json"
            (codex_home / "config.toml").write_text(
                "\n".join(
                    [
                        'model = "gpt-test"',
                        f'model_catalog_json = "{catalog_path}"',
                        "",
                        "[profiles.deepseek-pro]",
                        'model_provider = "deepseek"',
                        'model = "deepseek-v4-pro"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            _, profile_path = patch_codex_config(
                codex_home=codex_home,
                api_key="test-key",
                model="deepseek-v4-pro",
                provider="deepseek",
                profile="deepseek-pro",
                port=8877,
                catalog_path=catalog_path,
                auth_mode="direct",
            )

            base_config = (codex_home / "config.toml").read_text(encoding="utf-8")
            profile_config = profile_path.read_text(encoding="utf-8")

        self.assertIn('model = "gpt-test"', base_config)
        self.assertIn("[model_providers.deepseek]", base_config)
        self.assertNotIn("[profiles.deepseek-pro]", base_config)
        self.assertNotIn("model_catalog_json", base_config)
        self.assertIn('model_provider = "deepseek"', profile_config)
        self.assertIn('model = "deepseek-v4-pro"', profile_config)
        self.assertIn('service_tier = "default"', profile_config)
        self.assertIn(f'model_catalog_json = "{catalog_path}"', profile_config)

    def test_launch_agent_writes_route_and_auth_provider_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            proxy_path = home / ".codex" / "codex-deepseek-proxy" / "proxy.py"
            proxy_path.parent.mkdir(parents=True)
            proxy_path.touch()

            with patch("scripts.install.Path.home", return_value=home):
                plist_path = write_launch_agent(
                    proxy_path=proxy_path,
                    port=8877,
                    python_path="/usr/bin/python3",
                    thinking="auto",
                    model_routes=["qwen*=http://localhost:8000/v1/chat/completions"],
                    model_auth_providers=["qwen*=local-vllm"],
                )

            plist = plist_path.read_text(encoding="utf-8")

        self.assertIn("CODEX_DEEPSEEK_MODEL_ROUTES", plist)
        self.assertIn("CODEX_DEEPSEEK_MODEL_AUTH_PROVIDERS", plist)
        self.assertIn("qwen*=local-vllm", plist)
        self.assertIn("CODEX_DEEPSEEK_CODEX_CONFIG", plist)


if __name__ == "__main__":
    unittest.main()
