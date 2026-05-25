#!/usr/bin/env python3
"""Responses API compatibility proxy for Codex -> Chat Completions backends.

Codex custom model providers currently send OpenAI Responses API requests to
`{base_url}/responses`. DeepSeek and many local backends such as vLLM/GPUStack
expose OpenAI-compatible Chat Completions endpoints instead. This proxy accepts
Codex Responses requests locally, converts them to chat/completions, and streams
Responses-style events back.
"""

from __future__ import annotations

import json
import fnmatch
import os
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOST = os.environ.get("CODEX_DEEPSEEK_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CODEX_DEEPSEEK_PROXY_PORT", "8877"))
DEEPSEEK_CHAT_URL = os.environ.get(
    "DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions"
)
LOG_PATH = os.path.expanduser(
    os.environ.get("CODEX_DEEPSEEK_PROXY_LOG", "~/.codex/deepseek-responses-proxy.log")
)
THINKING = os.environ.get("CODEX_DEEPSEEK_THINKING", "disabled")
DEBUG_EVENTS = os.environ.get("CODEX_DEEPSEEK_DEBUG_EVENTS") in ("1", "true", "yes")


def parse_model_routes(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        routes: list[tuple[str, str]] = []
        for item in value.split(";"):
            if not item.strip() or "=" not in item:
                continue
            pattern, url = item.split("=", 1)
            routes.append((pattern.strip(), url.strip()))
        return routes

    if isinstance(parsed, dict):
        return [(str(pattern), str(url)) for pattern, url in parsed.items()]

    routes = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            pattern = item.get("model") or item.get("pattern")
            url = item.get("url") or item.get("chat_url")
            if pattern and url:
                routes.append((str(pattern), str(url)))
    return routes


MODEL_ROUTES = parse_model_routes(
    os.environ.get("CODEX_DEEPSEEK_MODEL_ROUTES")
    or os.environ.get("CODEX_CHAT_COMPLETIONS_MODEL_ROUTES")
)


def chat_url_for_model(
    model: str,
    routes: list[tuple[str, str]] | None = None,
    default_url: str | None = None,
) -> str:
    for pattern, url in routes if routes is not None else MODEL_ROUTES:
        if fnmatch.fnmatchcase(model, pattern):
            return url
    return default_url or DEEPSEEK_CHAT_URL


def log(message: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line)


def debug(message: str) -> None:
    if DEBUG_EVENTS:
        log(f"debug: {message}")


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append(json.dumps(part, ensure_ascii=False))
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text"):
            parts.append(part.get("text", ""))
        elif "text" in part:
            parts.append(str(part["text"]))
        elif part_type:
            parts.append(f"[{part_type}]")
    return "\n".join(piece for piece in parts if piece)


def responses_input_to_chat_messages(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = request_body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()

    for item in request_body.get("input") or []:
        if not isinstance(item, dict):
            flush_pending_tool_calls()
            messages.append({"role": "user", "content": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "message":
            flush_pending_tool_calls()
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            messages.append({"role": role, "content": flatten_content(item.get("content"))})
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            arguments = item.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "unknown",
                        "arguments": arguments,
                    },
                }
            )
            continue

        if item_type == "function_call_output":
            flush_pending_tool_calls()
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id"),
                    "content": output or "",
                }
            )
            continue

        if item_type == "reasoning":
            continue

        flush_pending_tool_calls()
        messages.append({"role": "user", "content": flatten_content(item)})

    flush_pending_tool_calls()
    return messages


def responses_tools_to_chat_tools(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    chat_tools: list[dict[str, Any]] = []
    for tool in request_body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return chat_tools


def message_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for index, message in enumerate(messages):
        entry: dict[str, Any] = {"i": index, "role": message.get("role")}
        if message.get("tool_calls"):
            entry["tool_call_ids"] = [call.get("id") for call in message["tool_calls"]]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message.get("tool_call_id")
        content = message.get("content")
        if isinstance(content, str) and content:
            entry["content_prefix"] = content[:80]
        summary.append(entry)
    return summary


def normalize_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provider-compatible chat history with one system message at index 0."""
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") != "system":
            non_system.append(message)
            continue

        content = message.get("content")
        if isinstance(content, str):
            if content:
                system_parts.append(content)
        elif content is not None:
            system_parts.append(json.dumps(content, ensure_ascii=False))

    if not system_parts:
        return messages

    normalized = [{"role": "system", "content": "\n\n".join(system_parts)}]
    normalized.extend(non_system)

    if len(system_parts) > 1 or messages[:1] != normalized[:1]:
        debug(
            "normalized system messages "
            f"parts={len(system_parts)} total_messages={len(normalized)}"
        )
    return normalized


def normalize_tool_call_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make chat history satisfy Chat Completions tool-call ordering rules."""
    normalized: list[dict[str, Any]] = []
    index = 0
    repairs = 0
    skip_indexes: set[int] = set()

    while index < len(messages):
        if index in skip_indexes:
            index += 1
            continue

        message = messages[index]

        if message.get("role") == "tool":
            repairs += 1
            normalized.append(
                {
                    "role": "user",
                    "content": (
                        "Tool output without matching assistant tool call was omitted "
                        f"by proxy. tool_call_id={message.get('tool_call_id')}"
                    ),
                }
            )
            index += 1
            continue

        normalized.append(message)

        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not tool_calls:
            index += 1
            continue

        required_ids = [call.get("id") for call in tool_calls if call.get("id")]
        found_by_id: dict[str, dict[str, Any]] = {}

        scan = index + 1
        while scan < len(messages) and messages[scan].get("role") == "tool":
            tool_id = messages[scan].get("tool_call_id")
            if tool_id in required_ids and tool_id not in found_by_id:
                found_by_id[tool_id] = messages[scan]
            scan += 1

        if len(found_by_id) < len(required_ids):
            later = scan
            while later < len(messages):
                candidate = messages[later]
                if candidate.get("role") == "assistant" and candidate.get("tool_calls"):
                    break
                if candidate.get("role") == "tool":
                    tool_id = candidate.get("tool_call_id")
                    if tool_id in required_ids and tool_id not in found_by_id:
                        found_by_id[tool_id] = candidate
                        skip_indexes.add(later)
                        repairs += 1
                later += 1

        for tool_id in required_ids:
            if tool_id in found_by_id:
                normalized.append(found_by_id[tool_id])
            else:
                repairs += 1
                normalized.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": (
                            "[tool output missing; inserted by proxy to keep chat "
                            "history valid]"
                        ),
                    }
                )

        index += 1
        while index < len(messages) and messages[index].get("role") == "tool":
            index += 1
        while index < len(messages) and index in skip_indexes:
            index += 1

    if repairs:
        log(
            "normalized tool history repairs="
            f"{repairs} summary={json.dumps(message_summary(normalized), ensure_ascii=False)}"
        )
    return normalized


def build_chat_request(request_body: dict[str, Any]) -> dict[str, Any]:
    messages = responses_input_to_chat_messages(request_body)
    messages = normalize_system_messages(messages)
    messages = normalize_tool_call_history(messages)
    chat_request: dict[str, Any] = {
        "model": request_body.get("model", "deepseek-v4-pro"),
        "messages": messages,
        "stream": True,
    }

    chat_tools = responses_tools_to_chat_tools(request_body)
    if chat_tools:
        chat_request["tools"] = chat_tools
        chat_request["tool_choice"] = request_body.get("tool_choice", "auto")

    max_output_tokens = request_body.get("max_output_tokens")
    if max_output_tokens:
        chat_request["max_tokens"] = max_output_tokens

    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = reasoning["effort"]
        chat_request["reasoning_effort"] = "max" if effort == "xhigh" else effort

    if THINKING != "omit":
        chat_request["thinking"] = {"type": THINKING}
    return chat_request


def base_response(
    request_body: dict[str, Any],
    response_id: str,
    status: str,
    output: list[dict[str, Any]],
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now if status == "completed" else None,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": request_body.get("model", "deepseek-v4-pro"),
        "output": output,
        "parallel_tool_calls": request_body.get("parallel_tool_calls", False),
        "previous_response_id": request_body.get("previous_response_id"),
        "reasoning": request_body.get("reasoning"),
        "store": request_body.get("store", False),
        "temperature": request_body.get("temperature"),
        "text": request_body.get("text", {"format": {"type": "text"}}),
        "tool_choice": request_body.get("tool_choice", "auto"),
        "tools": request_body.get("tools", []),
        "top_p": request_body.get("top_p"),
        "truncation": request_body.get("truncation"),
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata") or {},
    }


class SseWriter:
    def __init__(self, handler: BaseHTTPRequestHandler):
        self.handler = handler
        self.sequence_number = 1

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("type", event_type)
        payload.setdefault("sequence_number", self.sequence_number)
        self.sequence_number += 1
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        debug(f"send event={event_type} seq={self.sequence_number}")
        self.handler.wfile.write(f"event: {event_type}\n".encode("utf-8"))
        self.handler.wfile.write(b"data: ")
        self.handler.wfile.write(data)
        self.handler.wfile.write(b"\n\n")
        self.handler.wfile.flush()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            data = b"ok\n"
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/responses":
            self.send_error(404, "Only /responses is supported")
            return

        try:
            self.handle_responses()
        except BrokenPipeError:
            log("client disconnected")
        except Exception as exc:
            log(f"fatal: {exc}\n{traceback.format_exc()}")
            try:
                self.send_error(500, str(exc))
            except Exception:
                pass

    def handle_responses(self) -> None:
        length = int(self.headers.get("content-length", "0") or 0)
        request_body = json.loads(self.rfile.read(length).decode("utf-8"))
        auth_header = self.headers.get("authorization")
        if not auth_header:
            self.send_error(401, "Missing Authorization header")
            return

        chat_request = build_chat_request(request_body)
        upstream_chat_url = chat_url_for_model(str(chat_request.get("model") or ""))
        debug(
            "request "
            f"model={chat_request.get('model')} "
            f"upstream={upstream_chat_url} "
            f"messages={len(chat_request.get('messages') or [])} "
            f"tools={len(chat_request.get('tools') or [])} "
            f"max_tokens={chat_request.get('max_tokens')}"
        )
        upstream_request = urllib.request.Request(
            upstream_chat_url,
            data=json.dumps(chat_request, ensure_ascii=False).encode("utf-8"),
            headers={
                "authorization": auth_header,
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            method="POST",
        )

        response_id = f"resp_{uuid.uuid4().hex}"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        sse = SseWriter(self)
        sse.send(
            "response.created",
            {"response": base_response(request_body, response_id, "in_progress", [])},
        )
        sse.send(
            "response.in_progress",
            {"response": base_response(request_body, response_id, "in_progress", [])},
        )

        output: list[dict[str, Any]] = []
        text_state_holder: dict[str, Any] = {"state": None}
        tool_states: dict[int, dict[str, Any]] = {}

        try:
            with urllib.request.urlopen(upstream_request, timeout=600) as upstream:
                for raw_line in upstream:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    debug(
                        "chunk "
                        f"finish={choice.get('finish_reason')} "
                        f"content={len(delta.get('content') or '')} "
                        f"tool_calls={len(delta.get('tool_calls') or [])}"
                    )

                    self._stream_content_delta(delta, output, sse, text_state_holder)
                    self._stream_tool_deltas(delta, output, tool_states, sse)

                    if choice.get("finish_reason"):
                        break

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            message = f"Upstream HTTP {exc.code}: {detail}"
            log(f"{message} chat_summary={json.dumps(message_summary(chat_request['messages']), ensure_ascii=False)}")
            failed = base_response(request_body, response_id, "failed", output)
            failed["error"] = {"code": "upstream_error", "message": message}
            sse.send("response.failed", {"response": failed})
            return

        if text_state_holder["state"] is not None:
            self._finish_text_item(text_state_holder["state"], output, sse)
        self._finish_tool_items(tool_states, output, sse)
        completed = base_response(request_body, response_id, "completed", output)
        debug(
            "completed "
            f"output_items={len(output)} "
            f"text_chars={len(text_state_holder['state']['text']) if text_state_holder['state'] else 0} "
            f"tool_items={len(tool_states)}"
        )
        sse.send("response.completed", {"response": completed})

    @staticmethod
    def _stream_content_delta(
        delta: dict[str, Any],
        output: list[dict[str, Any]],
        sse: SseWriter,
        text_state_holder: dict[str, Any],
    ) -> None:
        content_delta = delta.get("content")
        if not content_delta:
            return

        state = text_state_holder.get("state")
        if state is None:
            item_id = f"msg_{uuid.uuid4().hex}"
            state = {"id": item_id, "text": ""}
            text_state_holder["state"] = state
            sse.send(
                "response.output_item.added",
                {
                    "output_index": len(output),
                    "item": {
                        "id": item_id,
                        "status": "in_progress",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
            sse.send(
                "response.content_part.added",
                {
                    "item_id": item_id,
                    "output_index": len(output),
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        state["text"] += content_delta
        sse.send(
            "response.output_text.delta",
            {
                "item_id": state["id"],
                "output_index": len(output),
                "content_index": 0,
                "delta": content_delta,
            },
        )

    @staticmethod
    def _stream_tool_deltas(
        delta: dict[str, Any],
        output: list[dict[str, Any]],
        tool_states: dict[int, dict[str, Any]],
        sse: SseWriter,
    ) -> None:
        for tool_delta in delta.get("tool_calls") or []:
            index = int(tool_delta.get("index", 0))
            state = tool_states.get(index)
            if state is None:
                call_id = tool_delta.get("id") or f"call_{uuid.uuid4().hex}"
                item_id = f"fc_{uuid.uuid4().hex}"
                state = {
                    "id": item_id,
                    "call_id": call_id,
                    "name": "",
                    "arguments": "",
                    "output_index": len(output) + len(tool_states),
                }
                tool_states[index] = state
                sse.send(
                    "response.output_item.added",
                    {
                        "output_index": state["output_index"],
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call_id,
                            "name": "",
                            "arguments": "",
                        },
                    },
                )

            function_delta = tool_delta.get("function") or {}
            if function_delta.get("name"):
                state["name"] += function_delta["name"]
            if function_delta.get("arguments"):
                state["arguments"] += function_delta["arguments"]
                sse.send(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": function_delta["arguments"],
                    },
                )

    @staticmethod
    def _finish_text_item(
        text_state: dict[str, str],
        output: list[dict[str, Any]],
        sse: SseWriter,
    ) -> None:
        text_item = {
            "id": text_state["id"],
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text_state["text"],
                    "annotations": [],
                }
            ],
        }
        output.append(text_item)
        sse.send(
            "response.output_text.done",
            {
                "item_id": text_state["id"],
                "output_index": 0,
                "content_index": 0,
                "text": text_state["text"],
            },
        )
        sse.send(
            "response.content_part.done",
            {
                "item_id": text_state["id"],
                "output_index": 0,
                "content_index": 0,
                "part": text_item["content"][0],
            },
        )
        sse.send("response.output_item.done", {"output_index": 0, "item": text_item})

    @staticmethod
    def _finish_tool_items(
        tool_states: dict[int, dict[str, Any]],
        output: list[dict[str, Any]],
        sse: SseWriter,
    ) -> None:
        for _, state in sorted(tool_states.items()):
            tool_item = {
                "id": state["id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"] or "{}",
            }
            output.append(tool_item)
            sse.send(
                "response.function_call_arguments.done",
                {
                    "item_id": state["id"],
                    "name": state["name"],
                    "output_index": state["output_index"],
                    "arguments": state["arguments"] or "{}",
                },
            )
            sse.send(
                "response.output_item.done",
                {"output_index": state["output_index"], "item": tool_item},
            )

    def log_message(self, format_string: str, *args: Any) -> None:
        log(f"{self.address_string()} {format_string % args}")


def main() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    log(
        f"listening on http://{HOST}:{PORT}, upstream={DEEPSEEK_CHAT_URL}, "
        f"routes={len(MODEL_ROUTES)}, thinking={THINKING}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
