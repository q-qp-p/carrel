"""Image-aware model registration, using synthetic inputs and isolated state."""
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from kb_pipeline import db
from kb_server import service


def response(payload, status=200):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(payload).encode()
    return r


def completion(text='A blue square.', protocol='openai', finish=None):
    if protocol == 'anthropic':
        return {'content': [{'type': 'text', 'text': text}], 'stop_reason': finish or 'end_turn'}
    return {'choices': [{'message': {'content': text}, 'finish_reason': finish or 'stop'}]}


@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    db.init_db(path)
    monkeypatch.setattr(service, 'settings', lambda: SimpleNamespace(state_db=path))
    with db.connect(path) as con:
        db.upsert_llm(con, name='Synthetic', base_url='http://models.invalid/v1',
                      api_key='original-key', model_id='original', concurrency=3)
    return path


def payload(**updates):
    return {'name': 'Synthetic', 'base_url': 'http://models.invalid/v1', 'model_id': 'vision',
            'api_key': 'probe-key', 'max_input_tokens': 8192, 'concurrency': 5, 'multimodal': True,
            **updates}


def row(path):
    with db.connect(path) as con:
        return dict(db.get_llm(con, 'Synthetic'))


@pytest.fixture
def endpoint():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, dict(self.headers), body))
            protocol = 'anthropic' if self.path.endswith('/messages') else 'openai'
            out = json.dumps(completion(protocol=protocol)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_save_really_sends_a_decodable_image_and_accepts_natural_reply(state, endpoint, protocol):
    url, calls = endpoint
    base = url + '/v1' if protocol == 'openai' else url
    saved = service.save_llm(payload(base_url=base, protocol=protocol))
    assert saved['multimodal'] is True and saved['concurrency'] == 5
    path, headers, body = calls[-1]
    assert path == ('/v1/chat/completions' if protocol == 'openai' else '/v1/messages')
    parts = body['messages'][0]['content']
    assert parts[0]['type'] == 'text' and 'attached image' in parts[0]['text']
    if protocol == 'openai':
        assert headers['Authorization'] == 'Bearer probe-key'
        image = parts[1]['image_url']['url']
        assert image.startswith('data:image/png;base64,')
        data = image.split(',', 1)[1]
    else:
        assert headers['x-api-key'] == 'probe-key'
        assert parts[1]['source']['media_type'] == 'image/png'
        data = parts[1]['source']['data']
    with Image.open(io.BytesIO(base64.b64decode(data))) as picture:
        picture.verify()
    # Saving again must probe again, retaining both the flag and the key if omitted.
    second = payload(base_url=base, protocol=protocol, api_key='')
    del second['multimodal']
    service.save_llm(second)
    assert len(calls) == 2
    assert isinstance(calls[-1][2]['messages'][0]['content'], list)
    assert 'A blue square.' not in json.dumps(row(state))
    assert 'probe-key' not in json.dumps(service.list_llms())
    with db.connect(state) as con:
        assert con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
        assert con.execute('SELECT COUNT(*) FROM graph_builds').fetchone()[0] == 0


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_text_mode_uses_the_existing_text_only_check(state, monkeypatch, protocol):
    post = Mock(return_value=response(completion('OK', protocol)))
    monkeypatch.setattr(service.requests, 'post', post)
    service.save_llm(payload(multimodal=False, protocol=protocol))
    assert isinstance(post.call_args.kwargs['json']['messages'][0]['content'], str)
    assert post.call_args.kwargs['allow_redirects'] is False
    assert row(state)['multimodal'] == 0


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
@pytest.mark.parametrize('text', ['', '   ', None])
def test_empty_image_reply_never_overwrites_configuration(state, monkeypatch, protocol, text):
    before = row(state)
    monkeypatch.setattr(service.requests, 'post', Mock(return_value=response(completion(text, protocol))))
    with pytest.raises(ValueError, match='no text response'):
        service.save_llm(payload(protocol=protocol))
    assert row(state) == before


@pytest.mark.parametrize('bad', [
    response({'error': 'Image input is not supported'}, 400),
    response({'error': 'Service unavailable'}, 503),
    response({}, 302),
    response({'choices': []}),
    response(completion(finish='length')),
    response(completion(finish='content_filter')),
    response(completion(protocol='anthropic', finish='max_tokens')),
    response(completion({'unexpected': 'object'})),
])
def test_failed_image_probe_preserves_all_saved_fields(state, monkeypatch, bad):
    before = row(state)
    post = Mock(return_value=bad)
    monkeypatch.setattr(service.requests, 'post', post)
    protocol = 'anthropic' if 'stop_reason' in bad.json() else 'openai'
    with pytest.raises(ValueError):
        service.save_llm(payload(protocol=protocol))
    assert row(state) == before
    assert post.call_count == 1  # Never fall back to a text-only request.


def test_timeout_and_invalid_json_leave_new_model_unsaved(state, monkeypatch):
    bad_json = requests.Response()
    bad_json.status_code = 200
    bad_json._content = b'not json'
    for result in [requests.Timeout(), bad_json]:
        post = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
        monkeypatch.setattr(service.requests, 'post', post)
        with pytest.raises(ValueError):
            service.save_llm(payload(name='New model'))
        with db.connect(state) as con:
            assert db.get_llm(con, 'New model') is None


def test_compatible_text_blocks_are_valid_image_output(state, monkeypatch):
    monkeypatch.setattr(service.requests, 'post', Mock(return_value=response(completion(
        [{'type': 'text', 'text': 'A square.'}]))))
    assert service.save_llm(payload())['multimodal'] is True


def test_save_api_rejects_failed_image_probe_without_changing_configuration(state, monkeypatch):
    from fastapi.testclient import TestClient
    from kb_server.main import create_app
    monkeypatch.delenv('KB_WEB_TOKEN', raising=False)
    before = row(state)
    post = Mock(return_value=response({'error': 'Images unsupported'}, 400))
    monkeypatch.setattr(service.requests, 'post', post)
    client = TestClient(create_app())
    failed = client.post('/api/llms', json=payload())
    assert failed.status_code == 422
    assert row(state) == before
    post.return_value = response(completion())
    passed = client.post('/api/llms', json=payload())
    assert passed.status_code == 200 and passed.json()['multimodal'] is True
