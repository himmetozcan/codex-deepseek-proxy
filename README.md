# Codex DeepSeek Proxy

Run DeepSeek models from Codex by adding a small local compatibility proxy.

Codex custom providers send requests to the OpenAI Responses API path:

```text
{base_url}/responses
```

DeepSeek exposes an OpenAI-compatible Chat Completions endpoint:

```text
https://api.deepseek.com/chat/completions
```

This project bridges the two APIs:

```text
Codex -> http://127.0.0.1:8877/responses -> local proxy -> DeepSeek chat/completions
```

The proxy also converts streaming events and normalizes tool-call history so Codex shell/tool calls can continue working.

## Status

This is an experimental compatibility layer, not an official Codex or DeepSeek integration.

It was built for:

- Codex CLI custom model providers that require Responses-style requests
- DeepSeek models exposed through Chat Completions
- Local OpenAI-compatible Chat Completions backends such as vLLM or GPUStack
- macOS users who want the proxy to run automatically through `launchd`

## Security

Do not commit real API keys.

The installer can write your DeepSeek API key into your local `~/.codex/config.toml` because Codex supports `experimental_bearer_token`. That local file should stay private and should not be pushed to GitHub.

For a safer setup, use `--auth-mode env` and export `DEEPSEEK_API_KEY` instead.

## Requirements

- macOS for the LaunchAgent installer
- Python 3.10+
- Codex CLI installed and configured
- A DeepSeek API key

## Quick Install On macOS

Clone the repo and run:

```bash
./scripts/install-macos.sh
```

The installer prompts for your DeepSeek API key and then:

- copies the proxy to `~/.codex/codex-deepseek-proxy/proxy.py`
- creates `~/.codex/deepseek_model_catalog.json`
- patches `~/.codex/config.toml`
- installs a macOS LaunchAgent
- starts the local proxy on `127.0.0.1:8877`

Then start Codex with:

```bash
codex --profile deepseek-pro
```

## Install With Environment Variable Auth

If you do not want the API key written into `config.toml`:

```bash
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"
./scripts/install-macos.sh --auth-mode env
codex --profile deepseek-pro
```

With this mode, Codex reads the key from `DEEPSEEK_API_KEY`.

## What Gets Added To Codex Config

The installer adds a custom model catalog path and DeepSeek provider/profile blocks:

```toml
model_catalog_json = "/Users/YOU/.codex/deepseek_model_catalog.json"

[model_providers.deepseek]
name = "DeepSeek"
base_url = "http://127.0.0.1:8877"
experimental_bearer_token = "<your-deepseek-api-key>"

[profiles.deepseek-pro]
model_provider = "deepseek"
model = "deepseek-v4-pro"
model_reasoning_effort = "high"
```

See [examples/config-snippet.toml](examples/config-snippet.toml).

The generated catalog contains only the custom DeepSeek model metadata needed by
Codex. It does not copy GPT model records from Codex's local cache, and it does
not claim a DeepSeek context window.

## macOS Service

The installer writes a LaunchAgent:

```text
~/Library/LaunchAgents/com.codex.deepseek-responses-proxy.plist
```

The service runs:

```text
~/.codex/codex-deepseek-proxy/proxy.py
```

Health check:

```bash
curl http://127.0.0.1:8877/health
```

Expected output:

```text
ok
```

## Manual Run

You can run the proxy without launchd:

```bash
python3 -m codex_deepseek_proxy.proxy
```

Or after package installation:

```bash
codex-deepseek-proxy
```

Useful environment variables:

```bash
export CODEX_DEEPSEEK_PROXY_HOST="127.0.0.1"
export CODEX_DEEPSEEK_PROXY_PORT="8877"
export DEEPSEEK_CHAT_URL="https://api.deepseek.com/chat/completions"
export CODEX_DEEPSEEK_MODEL_ROUTES='{"qwen*":"http://YOUR_VLLM_HOST:8000/v1/chat/completions"}'
export CODEX_DEEPSEEK_PROXY_LOG="$HOME/.codex/deepseek-responses-proxy.log"
export CODEX_DEEPSEEK_THINKING="disabled"
```

Set `CODEX_DEEPSEEK_THINKING=omit` if your upstream endpoint rejects the `thinking` field.

## Local vLLM Or GPUStack Backends

The same proxy service can route selected models to a local OpenAI-compatible
Chat Completions backend while all other models keep using the default
`DEEPSEEK_CHAT_URL`.

Example manual run:

```bash
export CODEX_DEEPSEEK_PROXY_PORT="8877"
export DEEPSEEK_CHAT_URL="https://api.deepseek.com/chat/completions"
export CODEX_DEEPSEEK_MODEL_ROUTES='{"qwen*":"http://YOUR_VLLM_HOST:8000/v1/chat/completions"}'
export CODEX_DEEPSEEK_THINKING="omit"
python3 -m codex_deepseek_proxy.proxy
```

Then configure Codex with a provider that points at the local proxy:

```toml
[model_providers.local-vllm]
name = "Local vLLM"
base_url = "http://127.0.0.1:8877"
env_key = "GPUSTACK_API_KEY"

[profiles.local-vllm]
model_provider = "local-vllm"
model = "qwen3.6-35b-a3b-fp8"
model_reasoning_effort = "medium"
```

See [examples/vllm-config-snippet.toml](examples/vllm-config-snippet.toml).

Some local backends reject multiple `system` messages. The proxy combines all
system messages into one first message before forwarding the request.

## Uninstall

Remove the LaunchAgent and installed proxy files:

```bash
./scripts/uninstall-macos.sh
```

Also remove the Codex config blocks:

```bash
./scripts/uninstall-macos.sh --remove-config
```

## Troubleshooting

### `unexpected status 404 Not Found: ... /responses`

Codex is still pointing directly at DeepSeek. Make sure the provider `base_url` is:

```toml
base_url = "http://127.0.0.1:8877"
```

### `Model metadata ... not found`

Codex does not know the custom model. The installer writes `~/.codex/deepseek_model_catalog.json` and sets:

```toml
model_catalog_json = "/Users/YOU/.codex/deepseek_model_catalog.json"
```

Restart Codex after changing this.

### `tool_calls must be followed by tool messages`

This is a Chat Completions history ordering rule. The proxy includes a normalizer for Codex tool-call history. If this still appears, check:

```bash
tail -n 80 ~/.codex/deepseek-responses-proxy.log
```

### Proxy Is Not Running

Check launchd:

```bash
launchctl print gui/$(id -u)/com.codex.deepseek-responses-proxy
```

Restart it:

```bash
launchctl kickstart -k gui/$(id -u)/com.codex.deepseek-responses-proxy
```

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run syntax check:

```bash
python3 -m py_compile codex_deepseek_proxy/proxy.py scripts/install.py scripts/uninstall.py
```

## License

MIT
