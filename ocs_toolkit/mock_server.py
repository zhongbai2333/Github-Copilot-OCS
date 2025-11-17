"""Tiny HTTP server that mimics an AnswererWrapper-friendly API."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Tuple

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
Logger = Callable[[str], None]


def create_mock_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    logger: Logger | None = None,
) -> HTTPServer:
    """Create the mock HTTP server bound to the given host/port."""

    log = logger or (lambda message: None)

    class MockQuestionHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802
            self._write_empty()

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming)
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            data = json.loads(payload or "{}")
            body = json.dumps(_build_response(data), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            log(f"Mock 命中: {data.get('title', '')}")

        def log_message(self, format: str, *args: Tuple[object, ...]) -> None:  # noqa: A003
            log(format % args)

        def _write_empty(self, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return HTTPServer((host, port), MockQuestionHandler)


def _build_response(payload: dict) -> dict:
    title = payload.get("title", "")
    if "1+2" in title:
        answers = [
            {"question": title, "answer": "3"},
            {"question": title.replace("1+2", "2+3"), "answer": "5"},
        ]
        return {"code": 1, "results": [[item["question"], item["answer"]] for item in answers]}
    return {"code": 0, "msg": "No matching records"}


def main() -> None:
    server = create_mock_server()
    print(f"Mock server listening on http://{DEFAULT_HOST}:{DEFAULT_PORT} ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
