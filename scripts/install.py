#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import html
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROFILE = "deepseek-pro"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_PORT = 8877
LABEL = "com.codex.deepseek-responses-proxy"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def deepseek_model_metadata(model: str) -> dict[str, Any]:
    return {
        "slug": model,
        "display_name": "DeepSeek V4 Pro",
        "description": "DeepSeek model accessed through a local Responses-to-Chat-Completions proxy.",
        "default_reasoning_level": "high",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Low reasoning"},
            {"effort": "medium", "description": "Medium reasoning"},
            {"effort": "high", "description": "High reasoning"},
            {"effort": "xhigh", "description": "Extra high reasoning"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "base_instructions": "You are Codex, a coding agent.",
        "model_messages": {
            "instructions_template": "You are Codex, a coding agent.\n\n{{ personality }}",
            "instructions_variables": {},
        },
        "supports_reasoning_summaries": False,
        "default_reasoning_summary": "none",
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
    }


def read_api_key(args: argparse.Namespace) -> str:
    if args.auth_mode == "env":
        return ""
    if args.api_key:
        return args.api_key
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    return getpass.getpass("DeepSeek API key: ").strip()


def install_proxy(codex_home: Path) -> Path:
    install_dir = codex_home / "codex-deepseek-proxy"
    install_dir.mkdir(parents=True, exist_ok=True)
    source = repo_root() / "codex_deepseek_proxy" / "proxy.py"
    target = install_dir / "proxy.py"
    shutil.copy2(source, target)
    target.chmod(0o755)
    return target


def write_model_catalog(codex_home: Path, model: str) -> Path:
    catalog_path = codex_home / "deepseek_model_catalog.json"
    output: dict[str, Any] = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "etag": None,
        "client_version": None,
        "models": [deepseek_model_metadata(model)],
    }
    catalog_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return catalog_path


def remove_table_block(lines: list[str], header: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == header:
            index += 1
            while index < len(lines) and not lines[index].startswith("["):
                index += 1
            continue
        output.append(lines[index])
        index += 1
    return output


def write_profile_config(
    codex_home: Path,
    profile: str,
    provider: str,
    model: str,
    catalog_path: Path,
) -> Path:
    profile_path = codex_home / f"{profile}.config.toml"
    profile_path.write_text(
        "\n".join(
            [
                f'model_provider = "{provider}"',
                f'model = "{model}"',
                'model_reasoning_effort = "high"',
                'service_tier = "default"',
                f'model_catalog_json = "{catalog_path}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return profile_path


def patch_codex_config(
    codex_home: Path,
    api_key: str,
    model: str,
    provider: str,
    profile: str,
    port: int,
    catalog_path: Path,
    auth_mode: str,
) -> tuple[Path, Path]:
    config_path = codex_home / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        original = config_path.read_text(encoding="utf-8")
    else:
        original = ""

    backup_path = config_path.with_suffix(f".toml.bak-{time.strftime('%Y%m%d%H%M%S')}")
    if original:
        backup_path.write_text(original, encoding="utf-8")

    lines = original.splitlines()
    lines = remove_table_block(lines, f"[model_providers.{provider}]")
    # Migrate the profile format used by Codex versions older than 0.134.0.
    lines = remove_table_block(lines, f"[profiles.{profile}]")
    legacy_catalog_line = f'model_catalog_json = "{catalog_path}"'
    lines = [line for line in lines if line.strip() != legacy_catalog_line]

    auth_line = (
        f'env_key = "DEEPSEEK_API_KEY"'
        if auth_mode == "env"
        else f'experimental_bearer_token = "{api_key}"'
    )
    lines.extend(
        [
            "",
            f"[model_providers.{provider}]",
            'name = "DeepSeek"',
            f'base_url = "http://127.0.0.1:{port}"',
            auth_line,
            "# Local proxy converts Codex Responses API calls to DeepSeek chat/completions.",
            "",
        ]
    )
    config_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    profile_path = write_profile_config(
        codex_home=codex_home,
        profile=profile,
        provider=provider,
        model=model,
        catalog_path=catalog_path,
    )
    return backup_path, profile_path


def write_launch_agent(
    proxy_path: Path,
    port: int,
    python_path: str,
    thinking: str,
    model_routes: list[str],
    model_auth_providers: list[str],
) -> Path:
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{LABEL}.plist"
    codex_home = proxy_path.parents[1]
    route_env = ""
    if model_routes:
        routes_value = html.escape(";".join(model_routes), quote=False)
        route_env = f"""    <key>CODEX_DEEPSEEK_MODEL_ROUTES</key>
    <string>{routes_value}</string>
"""
    auth_provider_env = ""
    if model_auth_providers:
        auth_providers_value = html.escape(";".join(model_auth_providers), quote=False)
        auth_provider_env = f"""    <key>CODEX_DEEPSEEK_MODEL_AUTH_PROVIDERS</key>
    <string>{auth_providers_value}</string>
"""
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python_path}</string>
    <string>{proxy_path}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CODEX_DEEPSEEK_PROXY_PORT</key>
    <string>{port}</string>
    <key>CODEX_DEEPSEEK_CODEX_CONFIG</key>
    <string>{codex_home}/config.toml</string>
    <key>CODEX_DEEPSEEK_THINKING</key>
    <string>{thinking}</string>
{route_env.rstrip()}
{auth_provider_env.rstrip()}
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{codex_home}/deepseek-responses-proxy.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>{codex_home}/deepseek-responses-proxy.stderr.log</string>
</dict>
</plist>
"""
    plist_path.write_text(plist)
    return plist_path


def launch_service(plist_path: Path) -> None:
    if platform.system() != "Darwin":
        print("Skipping launchd setup because this is not macOS.")
        return
    user_domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", user_domain, str(plist_path)], check=False)
    subprocess.run(["launchctl", "bootstrap", user_domain, str(plist_path)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"{user_domain}/{LABEL}"], check=True)


def health_check(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.read().decode("utf-8").strip() == "ok"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the Codex DeepSeek proxy.")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key for direct config mode.")
    parser.add_argument("--auth-mode", choices=["direct", "env"], default="direct")
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME", "~/.codex"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--python", default=shutil.which("python3") or sys.executable)
    parser.add_argument(
        "--thinking",
        default=os.environ.get("CODEX_DEEPSEEK_THINKING", "auto"),
        help=(
            "Value for CODEX_DEEPSEEK_THINKING. The default 'auto' enables "
            "DeepSeek thinking and omits the field for routed local backends."
        ),
    )
    parser.add_argument(
        "--model-route",
        action="append",
        default=[],
        metavar="PATTERN=URL",
        help="Route matching model names to a Chat Completions URL. Can be repeated.",
    )
    parser.add_argument(
        "--model-auth-provider",
        action="append",
        default=[],
        metavar="PATTERN=PROVIDER",
        help=(
            "Use a Codex model provider's configured bearer token for matching "
            "models. Can be repeated."
        ),
    )
    parser.add_argument("--no-launch-agent", action="store_true")
    args = parser.parse_args()

    api_key = read_api_key(args)
    if args.auth_mode == "direct" and not api_key:
        raise SystemExit("A DeepSeek API key is required for direct auth mode.")

    codex_home = Path(args.codex_home).expanduser()
    codex_home.mkdir(parents=True, exist_ok=True)
    proxy_path = install_proxy(codex_home)
    catalog_path = write_model_catalog(codex_home, args.model)
    backup_path, profile_path = patch_codex_config(
        codex_home=codex_home,
        api_key=api_key,
        model=args.model,
        provider=args.provider,
        profile=args.profile,
        port=args.port,
        catalog_path=catalog_path,
        auth_mode=args.auth_mode,
    )

    plist_path = None
    if not args.no_launch_agent:
        plist_path = write_launch_agent(
            proxy_path,
            args.port,
            args.python,
            args.thinking,
            args.model_route,
            args.model_auth_provider,
        )
        launch_service(plist_path)

    print(f"Installed proxy: {proxy_path}")
    print(f"Wrote model catalog: {catalog_path}")
    print(f"Patched Codex config: {codex_home / 'config.toml'}")
    print(f"Wrote Codex profile: {profile_path}")
    if backup_path.exists():
        print(f"Config backup: {backup_path}")
    if plist_path:
        print(f"LaunchAgent: {plist_path}")
        print(f"Health check: {'ok' if health_check(args.port) else 'failed'}")
    print(f"Use: codex --profile {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
