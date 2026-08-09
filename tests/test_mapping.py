import tempfile
import unittest
from pathlib import Path

from codex_deepseek_proxy import proxy
from codex_deepseek_proxy.proxy import (
    apply_thinking_controls,
    authorization_for_model,
    chat_url_for_model,
    normalize_system_messages,
    normalize_tool_call_history,
    parse_model_routes,
    responses_input_to_chat_messages,
)


class MappingTests(unittest.TestCase):
    def test_groups_responses_function_calls_before_outputs(self):
        request = {
            "instructions": "system",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "run commands"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "exec_command",
                    "arguments": '{"cmd":"date"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "/tmp",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": "Mon Jan 1",
                },
            ],
        }

        messages = responses_input_to_chat_messages(request)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(len(messages[2]["tool_calls"]), 2)
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_a")
        self.assertEqual(messages[4]["tool_call_id"], "call_b")

    def test_repairs_late_tool_outputs(self):
        messages = [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "y", "arguments": "{}"},
                    },
                ],
            },
            {"role": "assistant", "content": "interleaved text"},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
        ]

        fixed = normalize_tool_call_history(messages)

        self.assertEqual(fixed[1]["role"], "assistant")
        self.assertEqual(fixed[2]["role"], "tool")
        self.assertEqual(fixed[2]["tool_call_id"], "a")
        self.assertEqual(fixed[3]["role"], "tool")
        self.assertEqual(fixed[3]["tool_call_id"], "b")
        self.assertEqual(fixed[4]["role"], "assistant")

    def test_combines_system_messages_at_beginning(self):
        messages = [
            {"role": "system", "content": "base instructions"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "permissions"},
            {"role": "assistant", "content": "hi"},
        ]

        fixed = normalize_system_messages(messages)

        self.assertEqual(fixed[0]["role"], "system")
        self.assertEqual(fixed[0]["content"], "base instructions\n\npermissions")
        self.assertEqual([message["role"] for message in fixed], ["system", "user", "assistant"])

    def test_routes_models_to_configured_upstreams(self):
        routes = parse_model_routes(
            '{"qwen*":"http://127.0.0.1:8000/v1/chat/completions"}'
        )

        self.assertEqual(
            chat_url_for_model(
                "qwen3.6-35b-a3b-fp8",
                routes=routes,
                default_url="https://api.deepseek.com/chat/completions",
            ),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            chat_url_for_model(
                "deepseek-v4-pro",
                routes=routes,
                default_url="https://api.deepseek.com/chat/completions",
            ),
            "https://api.deepseek.com/chat/completions",
        )

    def test_uses_model_specific_provider_auth_from_codex_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model_providers.local-vllm]",
                        'experimental_bearer_token = "local-test-key"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            auth = authorization_for_model(
                "qwen3.6-35b-a3b-fp8",
                "Bearer deepseek-test-key",
                routes=[("qwen*", "local-vllm")],
                config_path=str(config_path),
            )

        self.assertEqual(auth, "Bearer local-test-key")

    def test_keeps_incoming_auth_without_model_specific_provider(self):
        auth = authorization_for_model(
            "deepseek-v4-pro",
            "Bearer deepseek-test-key",
            routes=[("qwen*", "local-vllm")],
        )

        self.assertEqual(auth, "Bearer deepseek-test-key")

    def test_auto_thinking_enables_deepseek_and_omits_local_routes(self):
        deepseek_request = {"reasoning": {"effort": "high"}}
        deepseek_chat = {"tool_choice": "auto"}
        apply_thinking_controls(
            deepseek_chat,
            deepseek_request,
            "https://api.deepseek.com/chat/completions",
        )

        self.assertEqual(deepseek_chat["thinking"], {"type": "enabled"})
        self.assertEqual(deepseek_chat["reasoning_effort"], "high")
        self.assertNotIn("tool_choice", deepseek_chat)

        local_chat = {}
        apply_thinking_controls(
            local_chat,
            deepseek_request,
            "http://example.local/v1/chat/completions",
        )

        self.assertNotIn("thinking", local_chat)
        self.assertNotIn("reasoning_effort", local_chat)

    def test_replays_reasoning_content_for_deepseek_tool_call_turns(self):
        proxy.REASONING_BY_TOOL_CALL["call_a"] = "hidden reasoning"
        try:
            request = {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_a",
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_a",
                        "output": "/tmp",
                    },
                ],
            }

            messages = responses_input_to_chat_messages(request)
        finally:
            proxy.REASONING_BY_TOOL_CALL.pop("call_a", None)

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["reasoning_content"], "hidden reasoning")
        self.assertEqual(messages[1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
