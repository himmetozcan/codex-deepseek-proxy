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
For DeepSeek thinking mode, it preserves the required `reasoning_content` for
tool-call turns and replays it in later requests.

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
- Python 3.11+
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
- adds the DeepSeek provider to `~/.codex/config.toml`
- creates `~/.codex/deepseek-pro.config.toml`
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

The installer adds the provider to the base config:

```toml
# ~/.codex/config.toml
[model_providers.deepseek]
name = "DeepSeek"
base_url = "http://127.0.0.1:8877"
experimental_bearer_token = "<your-deepseek-api-key>"
```

It writes the model selection to a separate profile file, which is the format
required by Codex 0.134.0 and later:

```toml
# ~/.codex/deepseek-pro.config.toml
model_provider = "deepseek"
model = "deepseek-v4-pro"
model_reasoning_effort = "high"
service_tier = "default"
model_catalog_json = "/Users/YOU/.codex/deepseek_model_catalog.json"
```

See [examples/config-snippet.toml](examples/config-snippet.toml) and
[examples/deepseek-pro.config.toml](examples/deepseek-pro.config.toml).

Selecting only `deepseek-v4-pro` from a model picker does not switch the active
provider. Start Codex with `codex --profile deepseek-pro` so the model and the
`deepseek` provider are selected together.

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
export CODEX_DEEPSEEK_CODEX_CONFIG="$HOME/.codex/config.toml"
export DEEPSEEK_CHAT_URL="https://api.deepseek.com/chat/completions"
export CODEX_DEEPSEEK_MODEL_ROUTES='{"qwen*":"http://YOUR_VLLM_HOST:8000/v1/chat/completions"}'
export CODEX_DEEPSEEK_MODEL_AUTH_PROVIDERS='{"qwen*":"local-vllm"}'
export CODEX_DEEPSEEK_PROXY_LOG="$HOME/.codex/deepseek-responses-proxy.log"
export CODEX_DEEPSEEK_THINKING="auto"
```

`CODEX_DEEPSEEK_THINKING=auto` enables thinking for the default DeepSeek
upstream and omits the `thinking` field for routed local backends. Set
`CODEX_DEEPSEEK_THINKING=omit` only if you intentionally want the proxy to omit
the field for every upstream.

When DeepSeek uses thinking mode with tool calls, the proxy stores the required
tool-call reasoning state locally in:

```text
~/.codex/deepseek-reasoning-state.json
```

## Local vLLM Or GPUStack Backends

The same proxy service can route selected models to a local OpenAI-compatible
Chat Completions backend while all other models keep using the default
`DEEPSEEK_CHAT_URL`.

Example manual run:

```bash
export CODEX_DEEPSEEK_PROXY_PORT="8877"
export DEEPSEEK_CHAT_URL="https://api.deepseek.com/chat/completions"
export CODEX_DEEPSEEK_MODEL_ROUTES='{"qwen*":"http://YOUR_VLLM_HOST:8000/v1/chat/completions"}'
export CODEX_DEEPSEEK_MODEL_AUTH_PROVIDERS='{"qwen*":"local-vllm"}'
export CODEX_DEEPSEEK_THINKING="auto"
python3 -m codex_deepseek_proxy.proxy
```

Then add a provider that points at the local proxy:

```toml
# ~/.codex/config.toml
[model_providers.local-vllm]
name = "Local vLLM"
base_url = "http://127.0.0.1:8877"
experimental_bearer_token = "<your-local-backend-api-key>"
```

Create a separate profile file:

```toml
# ~/.codex/local-vllm.config.toml
model_provider = "local-vllm"
model = "qwen3.6-35b-a3b-fp8"
model_reasoning_effort = "medium"
service_tier = "default"
model_catalog_json = "/Users/YOU/.codex/local_vllm_model_catalog.json"
```

See [examples/vllm-config-snippet.toml](examples/vllm-config-snippet.toml).

To expose DeepSeek and local models in one custom model picker, use one profile
whose catalog contains both models. Route local models to their Chat Completions
URL and tell the proxy which configured Codex provider owns their credentials:

```bash
export CODEX_DEEPSEEK_MODEL_ROUTES='{"qwen*":"http://YOUR_VLLM_HOST:8000/v1/chat/completions"}'
export CODEX_DEEPSEEK_MODEL_AUTH_PROVIDERS='{"qwen*":"local-vllm"}'
```

The second setting contains only a provider name. The proxy reads that provider's
`experimental_bearer_token` or `env_key` from the local Codex configuration and
does not copy the secret into the LaunchAgent. Provider selection is session-wide
in Codex, so GPT models authenticated through a ChatGPT account cannot be mixed
with these third-party models in the same picker.

When that provider uses `env_key`, the variable must also be available to the
proxy process. A macOS LaunchAgent does not automatically inherit variables from
your interactive shell; direct config auth avoids that launchd limitation.

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

### `reasoning_content ... must be passed back`

DeepSeek thinking mode requires `reasoning_content` to be preserved for
assistant messages that perform tool calls. This proxy captures that field from
DeepSeek streaming responses and replays it on later tool-call turns. If this
appears after upgrading from an older proxy version, start a new Codex session;
older sessions may contain tool-call history from before the proxy captured the
required reasoning state.

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
