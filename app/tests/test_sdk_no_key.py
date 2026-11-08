"""Configured no-auth endpoints must behave the same in probes and SDK clients."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from kb_pipeline.embedding.client import EmbeddingClient
from kb_pipeline.vision.vlm import _client_for as vision_client, _compatible_completion
from kb_pipeline.llm_compat import CompatibleRequest
from kb_search.channels import embed_question
from types import SimpleNamespace


@pytest.mark.parametrize("kind", ["text", "vision", "query"])
@pytest.mark.parametrize("key", ["", "synthetic"])
def test_optional_key_reaches_real_request_without_environment_fallback(
    kind, key, monkeypatch
):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(self.headers.get("Authorization"))
            body = json.dumps(
                {
                    "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                    "choices": [
                        {
                            "message": {"content": "OK", "role": "assistant"},
                            "finish_reason": "stop",
                            "index": 0,
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-be-used")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        if kind == "text":
            client = EmbeddingClient(
                base_url=url,
                api_key=key,
                model_id="synthetic",
                dim=2,
                batch_size=1,
                retry=1,
                sleep_seconds=0,
            )
            assert client.embed(["text"]) == [[1.0, 0.0]]
        elif kind == "query":
            settings = SimpleNamespace(
                embedding_base_url=url,
                embedding_api_key=key,
                embedding_model_id="synthetic",
                embedding_dim=2,
            )
            assert embed_question(settings, "text") == [1.0, 0.0]
        else:
            client = vision_client(url, key)
            body = {
                "model": "synthetic",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,synthetic"},
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
            }
            response = _compatible_completion(
                client, CompatibleRequest(url, "synthetic", "openai", body), api_key=key
            )
            assert response.choices[0].message.content == "OK"
        assert seen == (["Bearer synthetic"] if key else [None])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
