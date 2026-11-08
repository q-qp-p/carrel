"""Native Light data contracts; all persisted examples are synthetic."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import uuid

import numpy as np
import pytest

pytest.importorskip("lancedb")

from qdrant_client import QdrantClient
from kb_pipeline import db, discovery
from kb_pipeline.pipeline import parse_job, worker
from kb_pipeline.storage.light_data import LightChunks, LightKeywords
from kb_pipeline.storage.standard import StandardStorage
from kb_pipeline.vector.layout import VectorLayout
from test_scanner_worker import _parse_harness, _rescan


@pytest.fixture
def light(tmp_path):
    chunks = LightChunks(tmp_path / "storage")
    chunks.ensure("synthetic", VectorLayout(4, 4))
    return chunks, LightKeywords(chunks)


def payload(index, **updates):
    return dict(kb_id="synthetic", file_key=index // 3, doc_id=f"synthetic:{index // 3}",
                content_version="v1", is_active=True, chunk_uid=f"chunk-{index}", chunk_index=index % 3,
                block_id="b1", block_type="text", rel_path=f"folder/{index // 3}.md", filename=f"{index // 3}.md",
                text=f"Synthetic calibration item {index}.", **updates)


def test_exact_vectors_match_standard_with_null_visual_and_scopes(light):
    chunks, keywords = light
    rng = np.random.default_rng(902)
    ids = [str(uuid.uuid4()) for _ in range(40)]
    payloads = [payload(i) for i in range(40)]
    vectors = [{"text": rng.normal(size=4).tolist(), **({"visual": rng.normal(size=4).tolist()} if i % 3 == 0 else {})} for i in range(40)]
    chunks.write("synthetic", ids, vectors, payloads, max_bytes=4096)
    q = QdrantClient(":memory:")
    try:
        reference = StandardStorage(NS(), vector_client=q).chunks
        reference.ensure("synthetic", VectorLayout(4, 4))
        reference.write("synthetic", ids, vectors, payloads, max_bytes=4096)
        for visual in (False, True):
            for _ in range(12):
                query = rng.normal(size=4).tolist()
                for hints, excluded in ((None, None), ({"doc_ids": ["synthetic:0", "synthetic:1"]}, ["folder/1.md"]),
                                        ({"content_version": "missing"}, None)):
                    options = dict(limit=10, visual=visual, hints=hints, excluded=excluded)
                    a = chunks.vector_candidates("synthetic", query, **options)
                    b = reference.vector_candidates("synthetic", query, **options)
                    assert [row["point_id"] for row in a] == [row["point_id"] for row in b]
                    assert [row["score"] for row in a] == pytest.approx([row["score"] for row in b], abs=2e-6)
        assert len(chunks._rows("synthetic", "visual IS NULL")) == 26
        assert chunks.payloads("synthetic", ids) == dict(zip(ids, payloads))
    finally:
        q.close()


def test_mutation_context_metadata_and_reopen_match_standard(light, tmp_path):
    chunks, keywords = light
    ids = [str(uuid.uuid4()) for _ in range(6)]
    values = [payload(i) for i in range(6)]
    vectors = [{"text": [1., 0., 0., 0.]} for _ in ids]
    chunks.write("synthetic", ids, vectors, values, max_bytes=4096)
    for index, row in enumerate(values[:3]):
        assert [p["chunk_index"] for p in chunks.neighbors("synthetic", row, span=2)] == [i for i in range(3) if i != index]
    chunks.retire_previous("synthetic", kb_id="synthetic", file_key=0, active_content_version="v2")
    assert chunks.active_count("synthetic") == 3
    assert chunks.payloads("synthetic", ids[:3]) == {}
    assert len(chunks.payloads("synthetic", ids[:3], active_only=False)) == 3
    row = dict(collection="synthetic", kb_id="synthetic", file_key=0, content_version="v1", source_path="mirror/renamed.md",
               rel_path="renamed.md", filename="renamed.md", dir="", size=100, mtime=1)
    chunks.reactivate(row, point_ids=ids[:1])
    assert chunks.active_count("synthetic") == 4
    assert chunks.version_count(row) == 3
    assert chunks.payloads("synthetic", ids[:1])[ids[0]]["filename"] == "renamed.md"
    assert chunks.metadata("synthetic", [ids[0], ids[1], "missing"])[ids[1]]["active"] is False
    assert chunks.metadata("synthetic", ["missing"])["missing"] == {"active": False}
    assert chunks.retire_stale("synthetic", kb_id="synthetic", file_key=1, active_chunk_uids={"chunk-3"}) == 2
    reopened = LightChunks(tmp_path / "storage")
    assert reopened.active_count("synthetic") == 2
    reopened.check()
    reopened.deactivate(row)
    assert chunks.active_count("synthetic") == 1
    assert reopened.context("synthetic", doc_id="synthetic:1", chunk_from=0, chunk_to=10)[0]["chunk_uid"] == "chunk-3"


def test_keyword_updates_chinese_identifiers_and_scope(light):
    chunks, keyword = light
    ids = [str(uuid.uuid4()) for _ in range(3)]
    values = [payload(i * 3) for i in range(3)]
    values[0]["text"] = "项目预算包括设备采购与实施服务。 calibration_code ZX_417 returns 17."
    values[1]["text"] = "年度报告：设备采购金额为一百万元。"
    values[2]["text"] = "产品维护计划包含例行检查。"
    vectors = [{"text": [1., 0., 0., 0.]} for _ in ids]
    chunks.write("synthetic", ids, vectors, values, max_bytes=4096)
    keyword.sync_document("synthetic", "synthetic:0")
    assert {row["point_id"] for row in keyword.candidates("设备采购", "synthetic", limit=10)} == set(ids[:2])
    assert [row["point_id"] for row in keyword.candidates("calibration_code ZX_417", "synthetic", limit=10)] == ids[:1]
    assert not keyword.candidates("设备采购", "synthetic", limit=10, hints={"doc_ids": ["synthetic:2"]})
    assert [row["point_id"] for row in keyword.candidates("设备采购", "synthetic", limit=10, excluded=["folder/0.md"])] == ids[1:2]
    values[0]["text"] = "updated_code only"
    chunks.write("synthetic", ids[:1], vectors[:1], values[:1], max_bytes=4096)
    keyword.sync_document("synthetic", "synthetic:0")
    assert not keyword.candidates("calibration_code", "synthetic", limit=10)
    assert keyword.candidates("updated_code", "synthetic", limit=10)[0]["point_id"] == ids[0]
    chunks.deactivate(dict(collection="synthetic", kb_id="synthetic", file_key=1))
    assert not keyword.candidates("设备采购", "synthetic", limit=10)
    assert keyword.profile("updated_code 采购", ["synthetic", "absent"]) == {
        "terms": {"updated_code": {"synthetic": 1}, "采购": {}}, "sizes": {"synthetic": 2}}
    # Query punctuation and SQL-like document names are treated as data.
    assert not keyword.candidates("updated_code", "synthetic", limit=10, hints={"doc_ids": ["' OR true --"]})
    keyword.delete("synthetic")
    assert chunks.active_count("synthetic") == 2
    assert keyword.candidates("updated_code", "synthetic", limit=10)


def test_invalid_layout_write_is_atomic_and_no_visual_fabrication(light):
    chunks, keyword = light
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    with pytest.raises(ValueError, match="layout mismatch"):
        chunks.ensure("synthetic", VectorLayout(8, 4))
    with pytest.raises(ValueError, match="finite values"):
        chunks.write("synthetic", ids, [{"text": [1., 0., 0., 0.]}, {"text": [float("nan")] * 4}],
                     [payload(0), payload(1)], max_bytes=4096)
    assert chunks.active_count("synthetic") == 0
    chunks.write("synthetic", ids[:1], [{"text": [1., 0., 0., 0.]}], [payload(0)], max_bytes=4096)
    assert chunks.vector_candidates("synthetic", [1., 0., 0., 0.], limit=10, visual=True) == []
    with pytest.raises(ValueError, match="dimension"):
        chunks.vector_candidates("synthetic", [1., 0.], limit=10)


def test_keyword_mixed_script_and_partial_filename(light):
    chunks, keyword = light
    ids = [str(uuid.uuid4()) for _ in range(2)]
    values = [payload(0), payload(3)]
    values[0].update(filename="ZX417_calibration_notes.md", rel_path="guides/ZX417_calibration_notes.md",
                     text="接口ZX417返回检测结果。维护前先关闭阀门。")
    values[1].update(text="The unrelated schedule covers inspection dates.")
    chunks.write("synthetic", ids, [{"text": [1., 0., 0., 0.]} for _ in ids], values, max_bytes=4096)
    assert [r["point_id"] for r in keyword.candidates("ZX417检测结果", "synthetic", limit=10)] == ids[:1]
    assert [r["point_id"] for r in keyword.candidates("calibrat", "synthetic", limit=10)] == ids[:1]
    assert not keyword.candidates("calibrat", "synthetic", limit=10, hints={"doc_ids": ["synthetic:1"]})


def test_two_handles_concurrent_writes_preserve_all_rows(light, tmp_path):
    first, _ = light
    second = LightChunks(tmp_path / "storage")
    ids = [str(uuid.uuid4()) for _ in range(12)]
    def put(i):
        (first if i % 2 else second).write("synthetic", [ids[i]], [{"text": [1., 0., 0., 0.]}], [payload(i)], max_bytes=4096)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(put, range(12)))
    assert set(first.payloads("synthetic", ids)) == set(ids)
    assert second.active_count("synthetic") == 12


def test_realistic_dimensions_keep_missing_visual_and_valid_vectors(tmp_path):
    chunks = LightChunks(tmp_path / "storage")
    chunks.ensure("synthetic", VectorLayout(1024, 2048))
    ids = [str(uuid.uuid4()) for _ in range(3)]
    text = [0.] * 1023 + [1.]
    visual = [1.] + [0.] * 2047
    values = [payload(i) for i in range(3)]
    vectors = [{"text": text}, {"text": text, "visual": visual}, {"text": text}]
    chunks.write("synthetic", ids, vectors, values, max_bytes=4096)
    assert chunks.payloads("synthetic", ids) == dict(zip(ids, values))
    assert chunks.vectors("synthetic", ids) == dict.fromkeys(ids, text)
    assert [row["point_id"] for row in chunks.vector_candidates("synthetic", visual, limit=10, visual=True)] == ids[1:2]
    chunks.retire_stale("synthetic", kb_id="synthetic", file_key=0, active_chunk_uids={"chunk-1"})
    assert chunks.active_count("synthetic") == 1
    assert len(chunks._rows("synthetic", "visual IS NULL")) == 2


@pytest.mark.parametrize("filename,text", [
    ("notes.md", "# Operations\n\nORION valve pressure is 12 kPa.\n\n## Inspection\nInspect the seal before use."),
    ("code.md", "# Example\n\n```python\ndef calibration_code():\n    return 'ORION-27'\n```"),
    ("rates.csv", "item,value,unit\nORION,12,kPa\nVEGA,27,kPa\n"),
    ("guide.txt", "ORION uses 12 kPa. VEGA uses 27 kPa."),
])
def test_light_through_real_parse_worker_replace_and_empty(tmp_path, filename, text):
    original_parse = parse_job._parse_blocks
    with _parse_harness(str(tmp_path), filename=filename, text=text) as h, ExitStack() as stack:
        chunks = LightChunks(tmp_path / "storage")
        keywords = LightKeywords(chunks, h.settings)
        store = NS(chunks=chunks, keywords=keywords)
        h.settings.vlm_temperature = .1
        h.settings.vlm_top_p = .8
        h.settings.vlm_max_tokens = 64
        h.settings.vlm_structured_output = True
        stack.enter_context(patch.object(parse_job, "_parse_blocks", original_parse))
        stack.enter_context(patch.object(parse_job, "get_storage", return_value=store))
        stack.enter_context(patch.object(worker, "get_storage", return_value=store))
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
        ids = [row["point_id"] for row in db.active_chunk_refs(h.con, source.collection)]
        assert ids and "ORION" in "\n".join(row["text"] for row in chunks.payloads(source.collection, ids).values())
        assert keywords.candidates("ORION", source.collection, limit=10)
        matches = keywords.grep(source, ["ORION"])["results"][0]
        assert matches["total_chunks"] and matches["literal_true"]
        path = h.settings.mirror_root / "库A" / filename
        path.write_text(text.replace("ORION", "SIRIUS"))
        _rescan(h)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        assert not chunks.payloads(source.collection, ids)
        assert not keywords.candidates("ORION", source.collection, limit=10)
        assert keywords.candidates("SIRIUS", source.collection, limit=10)
        path.write_text("")
        _rescan(h, "empty")
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-empty:")
        assert chunks.active_count(source.collection) == 0
        assert not keywords.candidates("SIRIUS", source.collection, limit=10)
