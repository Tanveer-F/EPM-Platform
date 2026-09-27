"""Loopback-only HTTP API for local VS Code scoring and request-contract tests."""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from epm_platform.serving.scoring import (
    InferenceExecutionError,
    InferenceInputError,
    InferenceService,
)

_MAX_REQUEST_BYTES = 1_048_576


def create_server(service: InferenceService, host: str = "127.0.0.1", port: int = 8000):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("The local inference server may bind only to loopback.")

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, status: int, value: dict) -> None:
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._respond(404, {"error": "not_found"})
                return
            self._respond(200, {"status": "healthy"})

        def do_POST(self) -> None:
            if self.path != "/score":
                self._respond(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._respond(400, {"error": "invalid_content_length"})
                return
            if length < 1 or length > _MAX_REQUEST_BYTES:
                self._respond(413, {"error": "request_size_out_of_range"})
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self._respond(400, {"error": "incomplete_request_body"})
                return
            try:
                self._respond(200, service.score(body))
            except InferenceInputError as error:
                self._respond(400, {"error": error.code})
            except InferenceExecutionError as error:
                logging.getLogger("epm.inference").error(
                    "epm_local_api_failure error_class=%s", type(error).__name__
                )
                self._respond(500, {"error": "inference_failed"})

        def log_message(self, format: str, *args) -> None:
            logging.getLogger("epm.inference").info("epm_local_http " + format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("epm.inference").setLevel(logging.INFO)
    service = InferenceService.load(
        args.model_dir, Path(__file__).with_name("monitoring-reference.json")
    )
    server = create_server(service, args.host, args.port)
    try:
        print(f"Serving local inference on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
