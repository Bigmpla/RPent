"""Behavior checks for the opt-in API timing stream."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rpent.planner.http_timing import timing_client


def test_stream_timing_writes_metadata_without_payload(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (b": heartbeat\n\n", b'data: {"choices":[{"delta":',
                          b'{"content":"private-model-output"}}]}\n\n', b"data: [DONE]\n\n"):
                self.wfile.write(chunk)
                self.wfile.flush()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "timing.jsonl"

    async def run():
        async with timing_client(path) as client:
            async with client.stream(
                "POST", f"http://127.0.0.1:{server.server_port}/",
                json={"stream": True, "messages": [{"content": "private-user-input"}]},
                headers={"Authorization": "Bearer private-key"},
            ) as response:
                async for _ in response.aiter_bytes():
                    pass

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["status_code"] == 200
    assert result["stream_completed"]
    assert 0 <= result["send_start_s"] <= result["send_complete_s"]
    assert result["send_complete_s"] <= result["first_output_s"] <= result["end_s"]
    assert "private" not in lines[0]
