#!/usr/bin/env python3

import base64
import hashlib
import json
import os
import secrets
import socket
import ssl
import struct
import sys
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse


DEFAULT_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
DEFAULT_MODEL = "fun-asr-realtime"
DEFAULT_TIMEOUT = 30
DEFAULT_FINISH_GRACE_SECS = 1.0
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2


def write_stdout(event: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def write_stderr(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def normalize_transcript_text(text: str) -> str:
    return " ".join(text.split()).strip()


def combine_transcript(committed_text: str, current_text: str) -> str:
    committed = normalize_transcript_text(committed_text)
    current = normalize_transcript_text(current_text)

    if not committed:
        return current
    if not current:
        return committed
    if current == committed or current.startswith(committed):
        return current
    if committed.endswith(current):
        return committed
    return committed + " " + current


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name}.")
    return value


def get_optional_env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


def get_optional_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def get_optional_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return float(value)


def parse_language_hints(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def new_task_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ProviderState:
    task_id: str
    session_started: bool = False
    error: Optional[str] = None
    closed: bool = False
    server_finished: bool = False
    confirmed_text: str = ""
    latest_partial_text: str = ""
    last_final_text: str = ""

    def get_confirmed_text(self) -> str:
        return normalize_transcript_text(self.confirmed_text)

    def get_last_final_text(self) -> str:
        return normalize_transcript_text(self.last_final_text)

    def has_pending_partial(self) -> bool:
        return bool(normalize_transcript_text(self.latest_partial_text)) and (
            normalize_transcript_text(self.latest_partial_text) != self.get_confirmed_text()
        )


def emit_session_started(state: ProviderState, session_id: str) -> None:
    if state.session_started:
        return
    write_stdout({"type": "session_started", "session_id": session_id})
    state.session_started = True


def emit_partial_text(state: ProviderState, text: str) -> bool:
    partial_text = normalize_transcript_text(text)
    if not partial_text:
        return False
    if partial_text == normalize_transcript_text(state.latest_partial_text):
        return False
    state.latest_partial_text = partial_text
    write_stdout({"type": "partial", "text": partial_text})
    return True


def emit_final_text(state: ProviderState, text: str) -> bool:
    final_text = normalize_transcript_text(text)
    if not final_text or final_text == state.get_last_final_text():
        return False
    state.confirmed_text = final_text
    state.latest_partial_text = final_text
    state.last_final_text = final_text
    write_stdout({"type": "final", "text": final_text, "segment_final": True})
    return True


def emit_fallback_final(state: ProviderState) -> bool:
    if not state.has_pending_partial():
        return False
    return emit_final_text(state, state.latest_partial_text)


class WebSocketClient:
    def __init__(self, url: str, headers: Dict[str, str], timeout: int) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("WebSocket URL must use ws:// or wss://.")
        if not parsed.hostname:
            raise ValueError("WebSocket URL is missing a hostname.")

        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.scheme = parsed.scheme
        self.timeout = timeout
        self.headers = headers
        self._recv_buffer = b""
        self._closed = False
        self.socket = self._connect()

    def _connect(self) -> socket.socket:
        raw_sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        raw_sock.settimeout(self.timeout)

        if self.scheme == "wss":
            context = ssl.create_default_context()
            sock = context.wrap_socket(raw_sock, server_hostname=self.host)
        else:
            sock = raw_sock

        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        lines = [
            f"GET {self.path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        for name, value in self.headers.items():
            lines.append(f"{name}: {value}")
        request = "\r\n".join(lines) + "\r\n\r\n"
        sock.sendall(request.encode("utf-8"))

        response = self._read_http_response(sock)
        self._validate_handshake(response, key)
        return sock

    def _read_http_response(self, sock: socket.socket) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("WebSocket handshake failed: empty response.")
            data.extend(chunk)
            if len(data) > 65536:
                raise RuntimeError("WebSocket handshake failed: response too large.")
        return bytes(data)

    def _validate_handshake(self, response: bytes, key: str) -> None:
        header_blob = response.split(b"\r\n\r\n", 1)[0].decode(
            "utf-8", errors="replace"
        )
        lines = header_blob.split("\r\n")
        if not lines or "101" not in lines[0]:
            raise RuntimeError(
                f"WebSocket handshake failed: {lines[0] if lines else 'invalid response'}"
            )

        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        accept = headers.get("sec-websocket-accept")
        expected = base64.b64encode(
            hashlib.sha1((key + GUID).encode("utf-8")).digest()
        ).decode("ascii")
        if accept != expected:
            raise RuntimeError(
                "WebSocket handshake failed: invalid Sec-WebSocket-Accept header."
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self.socket.close()
        finally:
            self._closed = True

    def send_json(self, payload: Dict[str, Any]) -> None:
        self._send_frame(0x1, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def send_binary(self, payload: bytes) -> None:
        self._send_frame(0x2, payload)

    def recv_json(self) -> Optional[Dict[str, Any]]:
        fragments = bytearray()
        current_opcode: Optional[int] = None

        while True:
            frame = self._recv_frame()
            if frame is None:
                return None

            opcode, payload, fin = frame
            if opcode == 0x8:
                self._closed = True
                return None
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode not in {0x0, 0x1}:
                continue

            if opcode == 0x1:
                current_opcode = opcode
                fragments = bytearray(payload)
            else:
                if current_opcode is None:
                    continue
                fragments.extend(payload)

            if not fin:
                continue

            text = fragments.decode("utf-8", errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON message from DashScope: {exc}") from exc

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._closed:
            return

        first = 0x80 | (opcode & 0x0F)
        mask_key = secrets.token_bytes(4)
        length = len(payload)

        header = bytearray([first])
        if length < 126:
            header.append(0x80 | length)
        elif length < (1 << 16):
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))

        masked = bytes(payload[i] ^ mask_key[i % 4] for i in range(length))
        self.socket.sendall(bytes(header) + mask_key + masked)

    def _recv_frame(self) -> Optional[tuple[int, bytes, bool]]:
        header = self._recv_exact(2)
        if header is None:
            return None

        first, second = header
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F

        if length == 126:
            raw_length = self._recv_exact(2)
            if raw_length is None:
                return None
            length = struct.unpack("!H", raw_length)[0]
        elif length == 127:
            raw_length = self._recv_exact(8)
            if raw_length is None:
                return None
            length = struct.unpack("!Q", raw_length)[0]

        mask_key = b""
        if masked:
            mask_key = self._recv_exact(4)
            if mask_key is None:
                return None

        payload = self._recv_exact(length)
        if payload is None:
            return None

        if masked:
            payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(length))

        return opcode, payload, fin

    def _recv_exact(self, size: int) -> Optional[bytes]:
        while len(self._recv_buffer) < size:
            chunk = self.socket.recv(4096)
            if not chunk:
                if not self._recv_buffer and size > 0:
                    return None
                raise RuntimeError("WebSocket connection closed unexpectedly.")
            self._recv_buffer += chunk

        data = self._recv_buffer[:size]
        self._recv_buffer = self._recv_buffer[size:]
        return data


def build_run_task_event(
    task_id: str,
    model: str,
    language_hints: list[str],
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "format": "pcm",
        "sample_rate": 16000,
        "semantic_punctuation_enabled": False,
    }
    if language_hints:
        parameters["language_hints"] = language_hints

    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "parameters": parameters,
            "input": {},
        },
    }


def build_finish_task_event(task_id: str) -> Dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def get_header(message: Dict[str, Any]) -> Dict[str, Any]:
    header = message.get("header")
    return header if isinstance(header, dict) else {}


def get_event_name(message: Dict[str, Any]) -> str:
    header = get_header(message)
    event = header.get("event") or header.get("name") or message.get("event")
    return str(event or "").strip()


def extract_error_message(message: Dict[str, Any]) -> str:
    header = get_header(message)
    for container in (header, message.get("payload"), message):
        if not isinstance(container, dict):
            continue
        for key in ("message", "error_message", "errorMsg", "code"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        error = container.get("error")
        if isinstance(error, dict):
            nested = extract_error_message({"payload": error})
            if nested:
                return nested
        if isinstance(error, str) and error.strip():
            return error.strip()
    return "DashScope task failed."


def extract_sentence(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return None

    output = payload.get("output")
    if not isinstance(output, dict):
        return None

    sentence = output.get("sentence")
    if isinstance(sentence, dict):
        return sentence

    sentences = output.get("sentences")
    if isinstance(sentences, list):
        for item in reversed(sentences):
            if isinstance(item, dict):
                return item
    return None


def handle_result_generated(message: Dict[str, Any], state: ProviderState) -> None:
    sentence = extract_sentence(message)
    if sentence is None:
        return

    text = str(sentence.get("text") or sentence.get("sentence") or "")
    is_final_sentence = "end_time" in sentence and sentence.get("end_time") is not None
    visible_text = combine_transcript(state.confirmed_text, text)

    if is_final_sentence:
        if not normalize_transcript_text(visible_text):
            emit_fallback_final(state)
            return
        emit_final_text(state, visible_text)
        return

    emit_partial_text(state, visible_text)


def handle_server_message(message: Dict[str, Any], state: ProviderState) -> None:
    event_name = get_event_name(message)
    header = get_header(message)
    task_id = str(header.get("task_id") or state.task_id)

    if event_name == "task-started":
        emit_session_started(state, task_id)
        return

    if event_name == "result-generated":
        handle_result_generated(message, state)
        return

    if event_name == "task-finished":
        emit_fallback_final(state)
        state.server_finished = True
        return

    if event_name == "task-failed":
        error_message = extract_error_message(message)
        write_stdout({"type": "error", "message": error_message})
        state.error = error_message
        state.server_finished = True
        return

    if event_name == "error":
        error_message = extract_error_message(message)
        write_stdout({"type": "error", "message": error_message})
        state.error = error_message
        state.server_finished = True


def run() -> int:
    api_key = get_required_env("VINPUT_ASR_API_KEY")
    url = get_optional_env("VINPUT_ASR_URL", DEFAULT_URL)
    model = get_optional_env("VINPUT_ASR_MODEL", DEFAULT_MODEL)
    timeout = get_optional_int_env("VINPUT_ASR_TIMEOUT", DEFAULT_TIMEOUT)
    finish_grace_secs = get_optional_float_env(
        "VINPUT_ASR_FINISH_GRACE_SECS", DEFAULT_FINISH_GRACE_SECS
    )
    language_hints = parse_language_hints(get_optional_env("VINPUT_ASR_LANGUAGES"))
    task_id = new_task_id()

    client = WebSocketClient(url, {"Authorization": f"bearer {api_key}"}, timeout)
    client.send_json(build_run_task_event(task_id, model, language_hints))

    state = ProviderState(task_id=task_id)
    stop_event = threading.Event()
    task_started_event = threading.Event()

    def wait_for_task_started() -> bool:
        while not task_started_event.is_set():
            if stop_event.is_set():
                return False
            task_started_event.wait(timeout=0.05)
        return True

    def reader() -> None:
        try:
            while not stop_event.is_set():
                message = client.recv_json()
                if message is None:
                    if not stop_event.is_set() and not state.server_finished:
                        error_message = (
                            "DashScope connection closed before task finished."
                        )
                        state.error = error_message
                        write_stdout({"type": "error", "message": error_message})
                    break
                handle_server_message(message, state)
                if state.session_started:
                    task_started_event.set()
                if state.server_finished:
                    break
        except Exception as exc:
            if not stop_event.is_set():
                state.error = str(exc)
                write_stdout({"type": "error", "message": str(exc)})
        finally:
            stop_event.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    saw_finish = False
    try:
        for raw_line in sys.stdin:
            if stop_event.is_set():
                break

            line = raw_line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON input: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError("JSON input event must be an object.")

            event_type = str(event.get("type", "")).strip()
            if event_type == "audio":
                if not wait_for_task_started():
                    break
                audio_base64 = event.get("audio_base64")
                if not isinstance(audio_base64, str) or not audio_base64:
                    raise ValueError("audio event requires non-empty audio_base64.")
                try:
                    audio_bytes = base64.b64decode(audio_base64, validate=True)
                except ValueError as exc:
                    raise ValueError("audio_base64 is not valid base64.") from exc
                if not audio_bytes:
                    raise ValueError("audio event decoded to empty audio.")
                client.send_binary(audio_bytes)
                continue

            if event_type == "finish":
                if not wait_for_task_started():
                    break
                saw_finish = True
                client.send_json(build_finish_task_event(task_id))
                break

            if event_type == "cancel":
                stop_event.set()
                break

            raise ValueError(f"Unsupported event type: {event_type or '<missing>'}")
    finally:
        if saw_finish and not stop_event.is_set():
            thread.join(timeout=finish_grace_secs)
        if saw_finish:
            emit_fallback_final(state)
        stop_event.set()
        client.close()
        thread.join(timeout=1.0)
        if not state.closed:
            write_stdout({"type": "closed"})
            state.closed = True

    if state.error:
        return EXIT_RUNTIME_ERROR
    return 0


def main() -> int:
    try:
        return run()
    except ValueError as exc:
        write_stderr(str(exc))
        return EXIT_USAGE_ERROR
    except Exception as exc:
        write_stderr(str(exc))
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
