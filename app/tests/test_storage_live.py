"""Opt-in Standard integration against dedicated, disposable databases.

CARREL_STORAGE_INTEGRATION=1 enables this suite. Ports are deliberately different
from the application defaults. No production configuration or credentials load.
"""
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
from importlib.util import find_spec
import os

import pytest
from qdrant_client import QdrantClient

from kb_pipeline import db, discovery, search_fts
from kb_pipeline.graph import build, neo4j_import
from kb_pipeline.graph.extract import ExtractionResult
from kb_pipeline.graph.llm import LLMSpec, ChatClient
from kb_pipeline.pipeline import parse_job, worker
from kb_pipeline.storage import get_storage, standard
from test_scanner_worker import _parse_harness, _rescan
from _support import _schema_entry

pytestmark = pytest.mark.skipif(os.getenv("CARREL_STORAGE_INTEGRATION") != "1",
                                reason="requires disposable Standard databases")


@pytest.fixture
def live_db():
    q = QdrantClient(url="http://127.0.0.1:16333", check_compatibility=False)
    config = dict(qdrant_url="http://127.0.0.1:16333", qdrant_api_key=None,
                  opensearch_url="http://127.0.0.1:19200",
                  neo4j_uri="bolt://127.0.0.1:17687", neo4j_user="neo4j",
                  neo4j_password="synthetic-test-only")
    driver = neo4j_import.neo4j_driver(NS(**config))
    driver.verify_connectivity()
    q.get_collections()
    client = search_fts.client(config["opensearch_url"])
    client.cluster.health()
    # These ports are reserved for disposable databases; refuse nonempty ones.
    assert not q.get_collections().collections, "test Qdrant must be empty"
    with driver.session() as session:
        assert session.run("MATCH (n) RETURN count(n) AS n").single()["n"] == 0
    try:
        yield q, config
    finally:
        for alias in q.get_aliases().aliases:
            from qdrant_client import models
            q.update_collection_aliases([models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias.alias_name))])
        for collection in q.get_collections().collections:
            q.delete_collection(collection.name)
            search_fts.delete_collection(config["opensearch_url"], collection=collection.name)
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        driver.close()
        q.close()
        from kb_search import channels
        if channels._driver["driver"] is not None:
            channels._driver["driver"].close()
        channels._driver.update(key=None, driver=None)


def real_boundaries(stack, h, q, config, original_parse, original_sync, original_q):
    for key, value in config.items():
        setattr(h.settings, key, value)
    h.settings.vlm_temperature = 0.1
    h.settings.vlm_top_p = 0.8
    h.settings.vlm_max_tokens = 64
    h.settings.vlm_structured_output = True
    stack.enter_context(patch.object(parse_job, "_parse_blocks", original_parse))
    stack.enter_context(patch.object(standard, "vector_client", return_value=q))
    stack.enter_context(patch.object(search_fts, "sync_doc_from_qdrant", original_sync))
    for name, fn in original_q.items():
        stack.enter_context(patch.object(standard.qdrant, name, fn))


def capture_boundaries():
    return (parse_job._parse_blocks, search_fts.sync_doc_from_qdrant,
            {name: getattr(standard.qdrant, name) for name in (
                "collection_exists", "ensure_collection", "validate_collection_layout",
                "upsert_chunks", "mark_old_versions_inactive", "mark_stale_file_points_inactive")})


@pytest.mark.parametrize("filename,text", [
    ("notes.md", "# Calibration\n\nORION valve pressure is 12 kPa.\n\n## Inspection\nInspect every seal before use."),
    ("helper.md", "# Calibration helper\n\n```python\ndef code():\n    return 'ORION-27'\n```"),
    ("rates.csv", "item,value,unit\nORION,12,kPa\nVEGA,27,kPa\n"),
    ("guide.txt", "Synthetic calibration guide. ORION uses 12 kPa. VEGA uses 27 kPa."),
])
def test_live_parse_search_replace_withdraw(tmp_path, live_db, filename, text):
    q, config = live_db
    boundaries = capture_boundaries()
    with _parse_harness(str(tmp_path), filename=filename, text=text) as h, ExitStack() as stack:
        real_boundaries(stack, h, q, config, *boundaries)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
        store = get_storage(h.settings, vector_client=q)
        refs = db.active_chunk_refs(h.con, source.collection)
        ids = [row["point_id"] for row in refs]
        assert ids
        assert "ORION" in "\n".join(row["text"] for row in store.chunks.payloads(source.collection, ids).values())
        assert store.keywords.candidates("ORION", source.collection, limit=10)
        assert store.chunks.vector_candidates(source.collection, [1., 0., 0., 0.], limit=100)
        path = h.settings.mirror_root / "库A" / filename
        path.write_text(text.replace("ORION", "SIRIUS"))
        _rescan(h)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        assert not store.chunks.payloads(source.collection, ids)
        assert not store.keywords.candidates("ORION", source.collection, limit=10)
        assert store.keywords.candidates("SIRIUS", source.collection, limit=10)
        path.write_text("")
        _rescan(h, "empty")
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-empty:")
        assert store.chunks.active_count(source.collection) == 0
        assert not store.keywords.candidates("SIRIUS", source.collection, limit=10)


def test_live_graph_publication_readback_and_recovery(tmp_path, live_db, monkeypatch):
    q, config = live_db
    boundaries = capture_boundaries()
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
        real_boundaries(stack, h, q, config, *boundaries)
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
        store = get_storage(settings, vector_client=q)
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
        with patch.object(standard.StandardGraph, "activate", side_effect=RuntimeError("activation interrupted")):
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
        if find_spec("lancedb") is not None and find_spec("duckdb") is not None:
            from kb_pipeline.storage.light_data import LightChunks
            from kb_pipeline.storage.light_graph import LightGraphProjection, NODE_KINDS
            light = LightGraphProjection(LightChunks(tmp_path / "light"))
            driver = neo4j_import.neo4j_driver(settings)
            try:
                for version in ("v1", "v2"):
                    bundle = neo4j_import.load_bundle(build.graph_paths(settings, source, version).output_dir,
                                                     source=source, graph_version=version)
                    light.stage(source.kb_id, version, bundle)
                    light.activate(source.kb_id, version)
                    with driver.session() as session:
                        for kind in NODE_KINDS.values():
                            expected = [dict(row["n"]) for row in session.run(
                                f"MATCH (n:{kind} {{kb_id:$kb,graph_version:$version}}) RETURN n ORDER BY n.uid",
                                kb=source.kb_id, version=version)]
                            assert light.nodes(source.kb_id, kind) == expected
                        expected = session.run("""MATCH (r:Relation {kb_id:$kb,graph_version:$version})
                            <-[:EVIDENCES]-(:TextUnit)<-[:CONTRIBUTES_TO]-(c:QdrantChunkSnapshot)
                            RETURN DISTINCT r.uid,c.point_id,c.doc_id""", kb=source.kb_id, version=version)
                        assert set(light.evidence(source.kb_id)) == {tuple(row.values()) for row in expected}
            finally:
                driver.close()
