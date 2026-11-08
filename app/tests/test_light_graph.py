"""Native graph publication and evidence joins, with synthetic source assertions."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("duckdb")

from kb_pipeline.storage.light_data import LightChunks
from kb_pipeline.storage.light_graph import LightGraphProjection


def bundle(version):
    def node(kind, name, **props):
        return dict(uid=f"synthetic:{version}:{kind}:{name}", kb_id="synthetic",
                    graph_version=version, id=name, **props)
    doc = node("Document", "document", doc_id="source:7", doc_date="2026-03-01")
    unit = node("TextUnit", "unit", doc_id="source:7")
    chunk = node("QdrantChunkSnapshot", "chunk", doc_id="source:7", point_id="point-7")
    chunk.pop("id")
    entities = [node("Entity", name, title=name) for name in ("ORION", "pressure")]
    relation = node("Relation", "relation", type="has_property")
    facts = [
        node("Spec", "old", valid_from="2024-01-01", observed="2026-03-01", value="12", series_key="pressure"),
        node("Spec", "new", valid_from="2026-01-01", value="15", series_key="pressure", conflict_group="alternative"),
        node("Spec", "alternative", valid_from="2026-01-01", value="16", series_key="pressure", conflict_group="alternative"),
        node("Spec", "archived", doc_status="archived", value="11"),
        node("Spec", "interval", valid_from="2023-01-01", valid_until="2023-12-31", lapses=True, value="10"),
    ]
    return dict(documents=[doc], text_units=[unit], qdrant_chunks=[chunk], entities=entities,
        relations=[relation], specs=facts,
        document_edges=[dict(document_uid=doc["uid"], text_unit_uid=unit["uid"])],
        contributes_to_edges=[dict(chunk_uid=chunk["uid"], text_unit_uid=unit["uid"])],
        mentions_edges=[dict(text_unit_uid=unit["uid"], target_uid=e["uid"]) for e in entities],
        evidences_edges=[dict(text_unit_uid=unit["uid"], target_uid=relation["uid"])],
        relation_entity_edges=[dict(relation_uid=relation["uid"], source_uid=entities[0]["uid"],
            target_uid=entities[1]["uid"], related_props=dict(type="has_property"))],
        states_edges=[dict(text_unit_uid=unit["uid"], spec_uid=f["uid"]) for f in facts])


def projection(tmp_path):
    return LightGraphProjection(LightChunks(tmp_path / "storage"))


def test_publication_pins_reader_and_preserves_source_evidence(tmp_path):
    graph = projection(tmp_path)
    first = graph.stage("synthetic", "v1", bundle("v1"))
    assert graph.active_version("synthetic") is None
    graph.activate("synthetic", "v1")
    graph.stage("synthetic", "v2", bundle("v2"))
    assert graph.active_version("synthetic") == "v1"
    with graph.session("synthetic") as (version, con):
        assert version == "v1"
        graph.activate("synthetic", "v2")
        # A reader already in progress must not observe any part of v2.
        assert {row[0] for row in con.execute("SELECT json_extract_string(payload,'$.graph_version') FROM nodes").fetchall()} == {"v1"}
    reopened = projection(tmp_path)
    assert reopened.active_version("synthetic") == "v2"
    assert {e["title"] for e in reopened.nodes("synthetic", "Entity")} == {"ORION", "pressure"}
    assert reopened.neighbors("synthetic", "synthetic:v2:Entity:ORION") == [
        ("synthetic:v2:Relation:relation", "synthetic:v2:Entity:pressure", True, "has_property")]
    assert reopened.evidence("synthetic") == [("synthetic:v2:Relation:relation", "point-7", "source:7")]
    assert len(reopened.evidence("synthetic", kind="STATES")) == 5
    assert reopened.stage("synthetic", "v1", bundle("v1")) == first
    reopened.deactivate("synthetic")
    assert reopened.active_version("synthetic") is None
    assert len(reopened.nodes("synthetic", "Spec", version="v1")) == 5
    assert not list((tmp_path / "storage").glob("duckdb-*"))


def test_time_conflict_and_governance_use_existing_scope_semantics(tmp_path):
    graph = projection(tmp_path)
    graph.stage("synthetic", "v1", bundle("v1"))
    graph.activate("synthetic", "v1")
    ids = lambda **kw: {row["id"] for row in graph.facts("synthetic", **kw)}
    assert ids() == {"old", "new", "alternative", "interval"}
    assert ids(history=True) == {"old", "new", "alternative", "interval", "archived"}
    # The later document can substantiate a fact from an earlier period.
    assert ids(as_of="2025") == {"old", "archived"}
    assert ids(conflict_only=True) == {"new", "alternative"}
    assert ids(as_of="2025", conflict_only=True) == set()
    assert {f["value"] for f in graph.facts("synthetic", history=True) if f.get("series_key")} == {"12", "15", "16"}


@pytest.mark.parametrize("damage", ["missing_endpoint", "duplicate_node", "wrong_version", "published_overwrite"])
def test_rejected_stage_keeps_published_graph_intact(tmp_path, damage):
    graph = projection(tmp_path)
    graph.stage("synthetic", "v1", bundle("v1"))
    graph.activate("synthetic", "v1")
    version = "v1" if damage == "published_overwrite" else "v2"
    bad = bundle(version)
    if damage == "missing_endpoint": bad["entities"].pop()
    elif damage == "duplicate_node": bad["entities"].append(copy.deepcopy(bad["entities"][0]))
    elif damage == "wrong_version": bad["specs"][0]["graph_version"] = "unrelated"
    else: bad["specs"][0]["value"] = "changed"
    with pytest.raises(ValueError): graph.stage("synthetic", version, bad)
    assert graph.active_version("synthetic") == "v1"
    assert next(f for f in graph.facts("synthetic") if f["id"] == "old")["value"] == "12"


def test_killed_stage_can_resume_without_publishing_partial_graph(tmp_path):
    graph = projection(tmp_path)
    graph.stage("synthetic", "v1", bundle("v1"))
    graph.activate("synthetic", "v1")
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle("v2")))
    code = '''
import json,os,signal,sys
from pathlib import Path
from kb_pipeline.storage.light_data import LightChunks
from kb_pipeline.storage.light_graph import LightGraphProjection
data=LightChunks(Path(sys.argv[1]))
create=data.connection.create_table
def interrupted(name,*args,**kwargs):
    if name.endswith("_edges"): os.kill(os.getpid(),signal.SIGKILL)
    return create(name,*args,**kwargs)
data.connection.create_table=interrupted
LightGraphProjection(data).stage("synthetic","v2",json.loads(Path(sys.argv[2]).read_text()))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code, str(graph.data.root), str(source)],
                            env=env, capture_output=True, timeout=30)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()
    resumed = projection(tmp_path)
    assert resumed.active_version("synthetic") == "v1"
    with pytest.raises(KeyError): resumed.activate("synthetic", "v2")
    resumed.stage("synthetic", "v2", bundle("v2"))
    resumed.activate("synthetic", "v2")
    assert resumed.active_version("synthetic") == "v2"
    assert len(resumed.evidence("synthetic", kind="STATES")) == 5


def test_empty_graph_publishes_and_unknown_scope_does_not_fallback(tmp_path):
    graph = projection(tmp_path)
    graph.stage("synthetic", "v1", {})
    graph.activate("synthetic", "v1")
    assert graph.nodes("synthetic", "Entity") == []
    assert graph.evidence("synthetic") == []
    with pytest.raises(KeyError): graph.nodes("unrelated", "Entity")


def test_old_version_remains_immutable_after_new_version_is_published(tmp_path):
    graph = projection(tmp_path)
    for version in ("v1", "v2"):
        graph.stage("synthetic", version, bundle(version))
        graph.activate("synthetic", version)
    changed = bundle("v1")
    changed["specs"][0]["value"] = "changed"
    with pytest.raises(ValueError, match="cannot overwrite"):
        graph.stage("synthetic", "v1", changed)
    assert graph.active_version("synthetic") == "v2"
    assert next(f for f in graph.facts("synthetic", version="v1") if f["id"] == "old")["value"] == "12"
