#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


LABEL = "com.codex.deepseek-responses-proxy"
DEFAULT_MODEL = "deepseek-v4-pro"


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


def remove_top_level_assignment(
    lines: list[str],
    key: str,
    expected_value: str,
) -> list[str]:
    target = f'{key} = "{expected_value}"'
    table_index = next(
        (index for index, line in enumerate(lines) if line.startswith("[")),
        len(lines),
    )
    return [
        line
        for index, line in enumerate(lines)
        if index >= table_index or line.strip() != target
    ]


def remove_config(codex_home: Path, provider: str, profile: str, model: str) -> None:
    config_path = codex_home / "config.toml"
    if config_path.exists():
        lines = config_path.read_text(encoding="utf-8").splitlines()
        catalog_path = codex_home / "deepseek_model_catalog.json"
        lines = remove_top_level_assignment(
            lines, "model_catalog_json", str(catalog_path)
        )
        lines = remove_top_level_assignment(lines, "model_provider", provider)
        lines = remove_top_level_assignment(lines, "model", model)
        lines = remove_table_block(lines, f"[model_providers.{provider}]")
        lines = remove_table_block(lines, f"[profiles.{profile}]")
        config_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    (codex_home / f"{profile}.config.toml").unlink(missing_ok=True)


def unload_launch_agent(plist_path: Path) -> None:
    if platform.system() != "Darwin":
        return
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Uninstall the Codex DeepSeek proxy.")
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME", "~/.codex"))
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--profile", default="deepseek-pro")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--remove-config", action="store_true")
    args = parser.parse_args()

    codex_home = Path(args.codex_home).expanduser()
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    unload_launch_agent(plist_path)
    plist_path.unlink(missing_ok=True)

    shutil.rmtree(codex_home / "codex-deepseek-proxy", ignore_errors=True)
    (codex_home / "deepseek_model_catalog.json").unlink(missing_ok=True)

    if args.remove_config:
        remove_config(codex_home, args.provider, args.profile, args.model)

    print("Uninstalled launch agent and installed proxy files.")
    if not args.remove_config:
        print("Codex config was left intact. Re-run with --remove-config to remove it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
