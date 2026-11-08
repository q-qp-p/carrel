"""Model capabilities and request limits, using synthetic isolated state only."""
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kb_pipeline import db, discovery
from kb_pipeline.graph.build import graph_cache_fingerprint, resolve_llm_specs
from kb_pipeline.graph.llm import ChatClient, LLMSpec, cache_key
from kb_pipeline.pipeline import parse_job
from kb_pipeline.vision.vlm import caption_images_parallel
from kb_server import service


@pytest.fixture
def setup(tmp_path, monkeypatch):
    state = tmp_path / 'state.db'
    db.init_db(state)
    mirror = tmp_path / 'mirror'
    (mirror / 'documents').mkdir(parents=True)
    with db.connect(state) as con:
        source, _ = discovery.enroll(con, mirror, 'documents')
    cfg = SimpleNamespace(state_db=state, mirror_root=mirror, sources={source.kb_id: source})
    monkeypatch.setattr(service, 'settings', lambda: cfg)
    monkeypatch.setattr(service, '_probe_llm', Mock())
    return cfg, source


def payload(**updates):
    return {'name': 'Synthetic', 'base_url': 'http://model.invalid/v1', 'model_id': 'test-model',
            'api_key': 'test-key', 'max_input_tokens': 8192, **updates}


def test_save_list_and_edit_preserve_model_options_and_secret(setup):
    cfg, _ = setup
    saved = service.save_llm(payload(multimodal=True, concurrency=7))
    assert saved['multimodal'] is True and saved['concurrency'] == 7
    assert 'test-key' not in json.dumps(service.list_llms())
    edited = service.save_llm(payload(api_key=''))
    assert edited['multimodal'] is True and edited['concurrency'] == 7
    with db.connect(cfg.state_db) as con:
        assert db.get_llm(con, 'Synthetic')['api_key'] == 'test-key'
    new = service.save_llm(payload(name='Other'))
    assert new['multimodal'] is False and new['concurrency'] == 1


@pytest.mark.parametrize('updates', [
    {'multimodal': 1}, {'multimodal': 'true'}, {'multimodal': None},
    *({'concurrency': x} for x in (True, 0, -1, 257, 1.5, '4', None)),
])
def test_invalid_options_do_not_probe_or_overwrite(setup, updates):
    cfg, _ = setup
    service.save_llm(payload(multimodal=True, concurrency=7))
    service._probe_llm.reset_mock()
    with db.connect(cfg.state_db) as con:
        before = dict(db.get_llm(con, 'Synthetic'))
    with pytest.raises(ValueError):
        service.save_llm(payload(**updates))
    service._probe_llm.assert_not_called()
    with db.connect(cfg.state_db) as con:
        assert dict(db.get_llm(con, 'Synthetic')) == before


def test_active_visual_model_cannot_lose_capability_and_text_models_are_rejected(setup):
    cfg, source = setup
    service.save_llm(payload(multimodal=True, concurrency=4))
    service.save_llm(payload(name='Text only'))
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model': 'Synthetic'})
    with pytest.raises(ValueError, match='image descriptions'):
        service.save_llm(payload(multimodal=False))
    with pytest.raises(parse_job.NonRetryableParseError, match='not marked as multimodal'):
        parse_job._vlm_endpoint(cfg, replace(source, vlm_model='Text only'))


def test_each_new_task_reads_selected_model_options_without_cache_drift(setup):
    cfg, source = setup
    service.save_llm(payload(multimodal=True, concurrency=4))
    service.save_llm(payload(name='Summary', concurrency=9))
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model': 'Synthetic',
            'graph_llm': {'extract': 'Synthetic', 'summarize': 'Summary'}})
    source = replace(source, vlm_model='Synthetic')
    endpoint = parse_job._vlm_endpoint(cfg, source)
    specs = resolve_llm_specs(cfg, source, ('extract', 'summarize'))
    fingerprint = graph_cache_fingerprint(cfg, source)
    assert endpoint[3] == 4
    assert ChatClient(specs['extract']).workers == 4
    assert ChatClient(specs['summarize']).workers == 9
    service.save_llm(payload(multimodal=True, concurrency=2))
    assert endpoint[3] == 4  # An already-started document keeps its snapshot.
    assert parse_job._vlm_endpoint(cfg, source)[3] == 2
    assert resolve_llm_specs(cfg, source, ('extract',))['extract'].concurrency == 2
    assert graph_cache_fingerprint(cfg, source) == fingerprint
    assert cache_key(specs['extract'], [], {}) == cache_key(replace(specs['extract'], concurrency=2), [], {})


def test_legacy_options_are_adopted_once_from_usage_and_prior_limits(setup, monkeypatch):
    cfg, source = setup
    monkeypatch.setenv('VLM_CONCURRENCY', '3')
    monkeypatch.setenv('KB_GRAPH_LLM_CONCURRENCY', '11')
    (cfg.mirror_root / 'other').mkdir()
    with db.connect(cfg.state_db) as con:
        for name in ('Vision', 'Graph', 'Shared'):
            db.upsert_llm(con, name=name, base_url='http://model.invalid/v1', api_key='key', model_id=name)
        other, _ = discovery.enroll(con, cfg.mirror_root, 'other')
        discovery.set_config(con, source.kb_id, {'vlm_model': 'Vision', 'graph_llm': {'extract': 'Graph'}})
        discovery.set_config(con, other.kb_id, {'vlm_model': 'Shared', 'graph_llm': {'extract': 'Shared'}})
        con.execute('UPDATE llm_registry SET multimodal=NULL, concurrency=NULL')
    db.init_db(cfg.state_db)
    with db.connect(cfg.state_db) as con:
        rows = {r['name']: dict(r) for r in db.list_llms(con)}
    assert [(rows[n]['multimodal'], rows[n]['concurrency']) for n in ('Vision','Graph','Shared')] == [(1,3),(0,11),(1,3)]
    monkeypatch.setenv('VLM_CONCURRENCY', '7')
    db.init_db(cfg.state_db)
    with db.connect(cfg.state_db) as con:
        assert {r['name']: dict(r) for r in db.list_llms(con)} == rows


def measure_peak(fn):
    lock = threading.Lock()
    counts = {'active': 0, 'peak': 0}
    def work(*args, **kwargs):
        with lock:
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
        time.sleep(0.03)
        with lock:
            counts['active'] -= 1
        return {'ok': True}
    fn(work)
    return counts['peak']


@pytest.mark.parametrize('limit', [1, 3])
def test_graph_parallel_requests_follow_model_limit(limit):
    client = ChatClient(LLMSpec('synthetic', 'http://model.invalid/v1', '', 'm', concurrency=limit))
    assert measure_peak(lambda work: client.run_parallel(range(9), work)) == limit
    assert measure_peak(lambda work: client.run_parallel(range(9), work, workers=12)) == limit
    assert ChatClient(client.spec, workers=12).workers == limit
    assert ChatClient(client.spec, workers=1).workers == 1


@pytest.mark.parametrize('limit', [1, 3])
def test_visual_parallel_requests_follow_selected_model_limit(setup, tmp_path, monkeypatch, limit):
    cfg, source = setup
    service.save_llm(payload(multimodal=True, concurrency=limit))
    endpoint, key, model, workers = parse_job._vlm_endpoint(cfg, replace(source, vlm_model='Synthetic'))
    images = []
    for i in range(9):
        path = tmp_path / f'image-{i}.png'
        path.write_bytes(f'synthetic image {i}'.encode())
        images.append((str(i), path, None, None))
    def run(work):
        monkeypatch.setattr('kb_pipeline.vision.vlm.caption_image', work)
        result = caption_images_parallel(images, base_url=endpoint, api_key=key, model_id=model, concurrency=workers)
        assert len(result) == 9
    assert measure_peak(run) == limit
