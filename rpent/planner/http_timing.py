"""Opt-in timing for OpenAI-compatible HTTP requests.

Set RPENT_API_TIMING_FILE to an append-only JSONL path. Records metadata only.
Times are client-observed monotonic durations, not server-side GPU timings.
"""
import json
import os
import time
import uuid
from pathlib import Path

import httpx


class _TimedStream(httpx.AsyncByteStream):
    def __init__(self, stream, record, save):
        self.stream = stream
        self.record = record
        self.save = save
        self.buffer = b""

    def _observe(self, chunk):
        if self.record.get("first_output_ns") is not None:
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except (ValueError, UnicodeError):
                continue
            if not isinstance(data, dict):
                continue
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls", "function_call")):
                    self.record["first_output_ns"] = time.monotonic_ns()
                    return
        # Never retain arbitrary server response bodies.
        if len(self.buffer) > 65536:
            self.buffer = b""

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                self._observe(chunk)
                yield chunk
            self.record["stream_completed"] = True
        except BaseException as exc:
            self.record["error_type"] = type(exc).__name__
            raise
        finally:
            self.record["end_ns"] = time.monotonic_ns()
            self.save(self.record)

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            self.record.setdefault("end_ns", time.monotonic_ns())
            self.save(self.record)


def timing_client(path):
    """Construct a normal AsyncClient with metadata-only timing hooks."""
    target = Path(path).expanduser()

    def save(record):
        if record.get("_saved"):
            return
        record["_saved"] = True
        start = record["start_ns"]
        def seconds(name):
            value = record.get(name)
            return round((value - start) / 1e9, 6) if value is not None else None
        data = {
            "request_id": record["request_id"],
            "started_at_unix": record["started_at_unix"],
            "request_bytes": record["request_bytes"],
            "image_count": record.get("image_count"),
            "status_code": record.get("status_code"),
            "stream_requested": record.get("stream_requested"),
            "content_type": record.get("content_type"),
            "send_start_s": seconds("send_start_ns"),
            "send_complete_s": seconds("send_complete_ns"),
            "response_headers_s": seconds("headers_ns"),
            "first_output_s": seconds("first_output_ns"),
            "end_s": seconds("end_ns"),
            "stream_completed": record.get("stream_completed", False),
            "error_type": record.get("error_type"),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
        except OSError:
            # Timing must not prevent serving if the diagnostics path fails.
            import logging
            logging.getLogger(__name__).exception("Could not write API timing metadata")

    async def on_request(request):
        # SSE must be observed before HTTPX decompression; avoid compressed chunks.
        request.headers["Accept-Encoding"] = "identity"
        record = {
            "request_id": uuid.uuid4().hex,
            "start_ns": time.monotonic_ns(),
            "started_at_unix": time.time(),
            "request_bytes": int(request.headers.get("content-length", "0")),
        }
        try:
            payload = json.loads(request.content)
            record["stream_requested"] = payload.get("stream")
            # Only inspect message shapes; never persist user text or image data.
            record["image_count"] = sum(
                1
                for message in payload.get("messages", [])
                for item in (message.get("content") if isinstance(message.get("content"), list) else [])
                if isinstance(item, dict) and item.get("type") == "image_url"
            )
        except (ValueError, httpx.RequestNotRead):
            record["stream_requested"] = None
        request.extensions["rpent_timing"] = record
        previous = request.extensions.get("trace")

        async def trace(name, info):
            if previous:
                await previous(name, info)
            if name.endswith("send_request_body.started"):
                record["send_start_ns"] = time.monotonic_ns()
            elif name.endswith("send_request_body.complete"):
                record["send_complete_ns"] = time.monotonic_ns()
            elif name.endswith(".failed"):
                record["error_type"] = type(info.get("exception")).__name__
                record["end_ns"] = time.monotonic_ns()
                save(record)
        request.extensions["trace"] = trace

    async def on_response(response):
        record = response.request.extensions["rpent_timing"]
        record["headers_ns"] = time.monotonic_ns()
        record["status_code"] = response.status_code
        record["content_type"] = response.headers.get("content-type", "").split(";")[0]
        response.stream = _TimedStream(response.stream, record, save)

    return httpx.AsyncClient(event_hooks={"request": [on_request], "response": [on_response]})
