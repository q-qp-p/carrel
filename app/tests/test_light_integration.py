"""Light through real parsing/build/query paths; deterministic external model boundaries."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import pytest
pytest.importorskip("lancedb")
pytest.importorskip("duckdb")
from kb_pipeline import db, discovery
from kb_pipeline.graph import build
from kb_pipeline.graph.extract import ExtractionResult
from kb_pipeline.graph.llm import LLMSpec, ChatClient
from kb_pipeline.pipeline import parse_job, worker
from kb_pipeline.storage import get_storage
from kb_pipeline.storage.light import LightGraph
from test_scanner_worker import _parse_harness
from _support import _schema_entry

def test_light_graph_publication_readback_and_recovery(tmp_path, monkeypatch):
    original_parse = parse_job._parse_blocks
    text = "# Synthetic products\n\nORION operates beside VEGA. Both products require an annual seal inspection."
    spec = LLMSpec(name="fixture", base_url="http://unused", api_key="", model_id="fixture", protocol="openai")

    class Extractor:
        fingerprint = "storage-contract-v1"
        max_tokens = 100
        def __init__(self, *args, **kwargs): pass
        def extract(self, unit):
            return ExtractionResult(unit_id=unit.unit_id, entities=[
                {"name": name, "type": "product", "types": {"product": 1}, "mentions": 1, "descriptions": [unit.text]}
                for name in ("ORION", "VEGA")], relations=[
                    {"source": "ORION", "target": "VEGA", "type": "related_to", "description": unit.text, "weight": 1}],
                calls=1, stats={"unit_kind": "body"})

    monkeypatch.setenv("KB_GRAPH_VIEWS_NO_NARRATE", "1")
    monkeypatch.setattr(build, "GraphExtractor", Extractor)
    monkeypatch.setattr(build, "resolve_llm_specs", lambda *a, **kw: {"extract": spec, "summarize": spec})
    monkeypatch.setattr(build, "graph_embedder", lambda *a, **kw: NS(embed=lambda texts: [[1., 0., 0., 0.] for _ in texts]))
    monkeypatch.setattr(build, "_phase_retry_config", lambda: {"retries": 1, "backoff": 0})
    monkeypatch.setattr(ChatClient, "chat", lambda *a, **kw: "1. no\n2. no\n3. no")
    with _parse_harness(str(tmp_path), text=text) as h, ExitStack() as stack:
        stack.enter_context(patch.object(parse_job, "_parse_blocks", original_parse))
        h.settings.storage_backend = "light"
        h.settings.storage_root = tmp_path / "storage"
        h.settings.vlm_temperature = .1
        h.settings.vlm_top_p = .8
        h.settings.vlm_max_tokens = 64
        h.settings.vlm_structured_output = True
        settings = h.settings
        settings.runtime_dir = tmp_path / "runtime"
        settings.graph_work_dir = tmp_path / "graph"
        settings.corrections_dir = tmp_path / "corrections"
        settings.graph_neo4j_import_after_build = True
        settings.graph_neo4j_import_batch_size = 32
        settings.graph_llm_timeout_seconds = 2
        settings.graph_llm_concurrency = 1
        settings.graph_circuit_fails = 0
        discovery.set_config(h.con, h.kb_id, {
            "graph_enabled": True, "graph_schema_versions": [_schema_entry("s1", ["product"])],
            "graph_schema_active": "s1", "graph_entity_types": ["product"], "graph_parent_types": {"product": "entity"}})
        h.con.commit()
        assert worker.run_once(h.con, settings=settings).startswith("parse-done:")
        h.con.commit()
        source = discovery.enrolled_sources(settings.state_db, settings.mirror_root)[h.kb_id]
        settings.sources = {h.kb_id: source}
        store = get_storage(settings)
        def run(version, **options):
            return build.build_graph(settings, source_key=h.kb_id, source=source, graph_version=version,
                                     run_gc=False, **options)
        run("v1")
        rows = store.graph.entities(source)
        assert {r["title"] for r in rows["entities"]} == {"ORION", "VEGA"}
        neighbors = store.graph.neighbors(source, entity="ORION", entity_id=None)
        assert neighbors["found"] and neighbors["neighbors"]
        assert store.graph.stats(source.kb_id)[0] == "v1"
        old_aliases = store.graph.vector_aliases()
        with patch.object(LightGraph, "activate", side_effect=RuntimeError("activation interrupted")):
            with pytest.raises(RuntimeError, match="activation interrupted"):
                run("v2", incremental=True)
        assert store.graph.vector_aliases() == old_aliases
        assert store.graph.stats(source.kb_id)[0] == "v1"
        run("v2", incremental=True, allow_existing_graph_version=True)
        assert store.graph.stats(source.kb_id)[0] == "v2"
        before = json.loads(build.graph_paths(settings, source, "v1").graph_file.read_text())
        after = json.loads(build.graph_paths(settings, source, "v2").graph_file.read_text())
        assert {r["key"] for r in before["entities"]} == {r["key"] for r in after["entities"]}
        assert store.graph.neighbors(source, entity="ORION", entity_id=None)["neighbors"]
        assert store.graph.pages(source)["pages"]
        assert store.graph.facts(source)["found"]
        result = store.graph.query(source,"ORION",limit=10,hops=2,vector=[1.,0.,0.,0.])
        assert not result.get("skipped"), result
        assert result["entities"] and result["chunks"], result
        assert store.graph.neighborhoods(settings.sources,{source.kb_id:"v2"},"ORION",[],limit=2)
        assert store.catalog(ttl=0)[0]["has_graph"]
        # Exercise the real retrieval and HTTP contracts with the native adapter.
        # Only the external model response is fixed; storage and ranking are real.
        from kb_search import service as search_service, channels
        from kb_search.config import load_search_settings
        from kb_search.main import create_app
        from kb_pipeline.config import load_settings
        from dataclasses import fields, replace
        import os
        from fastapi.testclient import TestClient
        with patch.dict(os.environ, {'KB_RUNTIME_DIR':str(settings.runtime_dir), 'KB_STATE_DB':str(settings.state_db),
                                    'KB_MIRROR_ROOT':str(settings.mirror_root)}, clear=True):
            http_settings = load_settings(tmp_path/'empty.env')
        http_settings = replace(http_settings, **{field.name:getattr(settings,field.name)
            for field in fields(http_settings) if hasattr(settings,field.name)})
        http_store = get_storage(http_settings)
        search_settings = load_search_settings({'KB_SEARCH_TOKEN':'synthetic', 'KB_SEARCH_RERANK':'0',
            'KB_SEARCH_VISUAL':'0', 'KB_SEARCH_TEXT_DISCOVERY':'true'})
        with patch.object(search_service, 'runtime', return_value=(http_settings,search_settings,http_store)), \
             patch.object(channels, 'embed_question', return_value=[1.,0.,0.,0.]):
            client = TestClient(create_app())
            headers = {'Authorization':'Bearer synthetic'}
            health = client.get('/health',headers=headers).json()
            assert health['storage_backend']=='light'
            assert health['checks']['lancedb']=='ok' and 'qdrant' not in health['checks']
            response = client.post('/search', json={'question':'ORION', 'kbs':[source.kb_id]},headers=headers)
            assert response.status_code == 200, response.text
            answer = response.json()
            assert answer['sources'] and 'ORION' in json.dumps(answer['sources']), answer
            for route, body in (('/graph/entities',{}),('/graph/neighbors',{'entity':'ORION'}),
                                ('/graph/facts',{'subject':'ORION'}),('/graph/pages',{})):
                response = client.post(route,json={'kb_id':source.kb_id,**body},headers=headers)
                assert response.status_code == 200, (route,response.text)
        # Publication seals the vectors of every published version, not only active.
        name = next(iter(old_aliases.values()))
        with pytest.raises(ValueError, match='immutable'):
            store.graph.vectors.delete_except(name, [])
        store.graph.activate(source=source,graph_version="v1")
        assert store.graph.stats(source.kb_id)[0] == "v1"
        gc = build.gc_graph_versions(http_settings,source,storage=http_store,graph_version="v1",keep_latest=1)
        assert not gc["errors"] and store.graph.stats(source.kb_id)[0] == "v1"
        assert all("v2" not in n for n in store.graph.vector_collections())
        store.graph.drop(source.kb_id,source.collection)
        assert store.graph.stats(source.kb_id) == (None,{})
        assert not store.graph.vector_collections()
        assert store.chunks.active_count(source.collection) > 0
