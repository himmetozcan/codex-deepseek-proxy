import unittest

from codex_deepseek_proxy.proxy import (
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


if __name__ == "__main__":
    unittest.main()
