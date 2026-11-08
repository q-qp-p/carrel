"""Global endpoint settings use synthetic requests and isolated state only."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from kb_pipeline import config, db
from kb_pipeline.model_endpoints import CONFIG_KEY, apply_overrides, endpoint_values, read_overrides
from kb_server import model_endpoints as endpoints


@pytest.fixture
def setup(tmp_path, monkeypatch):
    env = tmp_path / "settings.env"
    env.write_text("")
    monkeypatch.setenv("KB_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("KB_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("KB_MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setenv("KB_GRAPH_WORK_DIR", str(tmp_path / "graph"))
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "synthetic-embed")
    monkeypatch.setenv("EMBEDDING_DIM", "3")
    monkeypatch.setenv("EMBEDDING_API_KEY", "synthetic-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://model.invalid/v1")
    monkeypatch.setenv("EMBEDDING_BATCH", "20")
    monkeypatch.setenv("EMBEDDING_CONCURRENCY", "1")
    monkeypatch.setenv("VISUAL_EMBEDDING_BATCH", "1")
    monkeypatch.setenv("VISUAL_EMBEDDING_CONCURRENCY", "2")
    monkeypatch.setenv("VISUAL_EMBEDDING_BASE_URL", "http://visual.invalid/v1")
    monkeypatch.setenv("VISUAL_EMBEDDING_MODEL_ID", "synthetic-visual")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "4")
    monkeypatch.setenv("RERANKER_BASE_URL", "http://rank.invalid/v1")
    monkeypatch.setenv("RERANKER_MODEL_ID", "synthetic-rank")
    monkeypatch.setenv("RERANKER_API_KEY", "synthetic-rank-key")
    monkeypatch.setenv("VISUAL_RERANKER_BASE_URL", "http://vrank.invalid/v1")
    cfg = config.load_settings(env)
    db.init_db(cfg.state_db)
    monkeypatch.setattr(endpoints, "load_settings", lambda: config.load_settings(env))
    monkeypatch.setattr(endpoints, "_probe", lambda role, row: row)
    return cfg, env


def body(role, **updates):
    row = next(x for x in endpoints.get_endpoints()["endpoints"] if x["role"] == role)
    return {k: v for k, v in row.items() if k not in {"role", "has_api_key", "validated"}} | updates


def test_current_configuration_is_displayed_without_secrets_or_writes(setup):
    cfg, _ = setup
    data = endpoints.get_endpoints()
    assert len(data["endpoints"]) == 4
    assert data["endpoints"][0]["model_id"] == "synthetic-embed"
    assert data["endpoints"][0]["has_api_key"] is True
    assert "synthetic-secret" not in json.dumps(data)
    assert read_overrides(cfg.state_db) == {}


def test_save_reloads_for_pipeline_and_preserves_other_roles_and_key(setup):
    cfg, env = setup
    original = endpoint_values(cfg)
    saved = endpoints.save_endpoint("embedding", body("embedding", base_url="http://model.invalid/new/v1", api_key=""))
    assert saved["base_url"].endswith("/new/v1")
    loaded = config.load_settings(env)
    assert loaded.embedding_base_url == saved["base_url"]
    assert loaded.embedding_api_key == "synthetic-secret"
    assert loaded.embedding_batch == cfg.embedding_batch
    assert endpoint_values(loaded)["visual_embedding"] == original["visual_embedding"]
    assert apply_overrides(cfg).embedding_base_url == saved["base_url"]  # next job, same worker
    assert "api_key" not in saved


def test_disabling_and_reenabling_reranker_preserves_endpoint(setup):
    cfg, env = setup
    off = endpoints.save_endpoint("reranker", body("reranker", enabled=False))
    assert off["base_url"] == "http://rank.invalid/v1"
    assert config.load_settings(env).reranker_base_url == ""
    on = endpoints.save_endpoint("reranker", body("reranker", enabled=True))
    assert config.load_settings(env).reranker_base_url == on["base_url"]


def test_visual_reranker_model_and_key_reach_runtime(setup):
    _, env = setup
    endpoints.save_endpoint("visual_reranker", body("visual_reranker", model_id="visual-rank", api_key="new-key"))
    loaded = config.load_settings(env)
    assert loaded.visual_reranker_model_id == "visual-rank"
    assert loaded.visual_reranker_api_key == "new-key"


def test_visual_embedding_can_be_disabled_separately(setup):
    cfg, env = setup
    endpoints.save_endpoint("visual_embedding", body("visual_embedding", enabled=False))
    loaded = config.load_settings(env)
    assert loaded.visual_embedding_enabled is False
    assert loaded.embedding_model_id == cfg.embedding_model_id


def test_new_host_does_not_silently_receive_existing_key(setup):
    cfg, env = setup
    with pytest.raises(ValueError, match="Re-enter or clear"):
        endpoints.save_endpoint("embedding", body("embedding", base_url="https://other.invalid/v1"))
    assert read_overrides(cfg.state_db) == {}
    endpoints.save_endpoint("embedding", body("embedding", base_url="https://other.invalid/v1", clear_api_key=True))
    assert config.load_settings(env).embedding_api_key == ""


@pytest.mark.parametrize("updates", [
    {"dim": True}, {"dim": 0}, {"dim": 1.5}, {"dim": 65537}, {"enabled": "false"},
    {"base_url": "file:///tmp/example"}, {"base_url": "https://u:p@example.invalid/v1"},
    {"base_url": "http://example.invalid/v1?key=secret"}, {"base_url": ""},
    {"model_id": ""}, {"api_key": 7}, {"enabled": False}, {"unknown": "x"},
    *({"batch_size": value} for value in (True, 0, 1.5, 1025, "2")),
    *({"concurrency": value} for value in (True, 0, 1.5, 33, "2")),
])
def test_invalid_settings_do_not_mutate_saved_configuration(setup, updates):
    cfg, _ = setup
    with pytest.raises(ValueError):
        endpoints.save_endpoint("embedding", body("embedding", **updates))
    assert read_overrides(cfg.state_db) == {}


def test_stale_form_and_probe_failure_preserve_saved_configuration(setup, monkeypatch):
    cfg, _ = setup
    stale = body("reranker")
    endpoints.save_endpoint("reranker", body("reranker", model_id="rank-new"))
    before = read_overrides(cfg.state_db)
    with pytest.raises(ValueError, match="refresh"):
        endpoints.save_endpoint("reranker", stale)
    monkeypatch.setattr(endpoints, "_probe", Mock(side_effect=ValueError("unavailable")))
    with pytest.raises(ValueError, match="unavailable"):
        endpoints.save_endpoint("reranker", body("reranker", model_id="rank-unreachable"))
    assert read_overrides(cfg.state_db) == before


def test_concurrent_save_during_probe_is_not_overwritten(setup, monkeypatch):
    cfg, _ = setup
    original = endpoints._probe
    def interleave(role, row):
        monkeypatch.setattr(endpoints, "_probe", original)
        endpoints.save_endpoint("visual_reranker", body("visual_reranker", model_id="other-save"))
        return row
    monkeypatch.setattr(endpoints, "_probe", interleave)
    with pytest.raises(ValueError, match="refresh"):
        endpoints.save_endpoint("reranker", body("reranker", model_id="lost-update"))
    assert set(read_overrides(cfg.state_db)) == {"visual_reranker"}


def test_existing_indexes_reject_new_vector_space_even_at_same_dimension(setup):
    cfg, _ = setup
    with db.connect(cfg.state_db) as con:
        con.execute("INSERT INTO chunks(chunk_uid,file_id,content_version,chunk_index,point_id,collection,status,created_at) "
                    "VALUES ('c','f','v',0,'p','kb','active',1)")
    for updates in ({"model_id": "different-model"}, {"dim": 7}):
        with pytest.raises(ValueError, match="indexes exist"):
            endpoints.save_endpoint("embedding", body("embedding", **updates))
    endpoints.save_endpoint("embedding", body("embedding", api_key="rotated-key"))
    with db.connect(cfg.state_db) as con:
        assert con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1


def test_running_graph_blocks_embedding_change_but_not_reranker(setup):
    cfg, _ = setup
    with db.connect(cfg.state_db) as con:
        con.execute("INSERT INTO graph_builds(graph_build_id,source_key,kb_id,source_collection,graph_version,status,started_at) "
                    "VALUES ('g','s','k','c','v','running',1)")
    with pytest.raises(ValueError, match="tasks to finish"):
        endpoints.save_endpoint("embedding", body("embedding", api_key="changed"))
    endpoints.save_endpoint("reranker", body("reranker", model_id="new-rank"))


def test_search_refresh_uses_saved_endpoint_without_restarting(setup, monkeypatch):
    from kb_search import service
    _, env = setup
    fresh = {"settings": None, "search": None, "store": None, "at": 0.0, "explicit": None}
    monkeypatch.setattr(service, "_state", {**service._state, **fresh})
    monkeypatch.setattr(service, "load_settings", lambda: config.load_settings(env))
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: SimpleNamespace(**kw))
    before, _, store = service.runtime()
    q = store._q
    endpoints.save_endpoint("reranker", body("reranker", model_id="new-rank"))
    assert service.runtime()[0].reranker_model_id == before.reranker_model_id
    service._state["at"] = 0
    assert service.runtime()[0].reranker_model_id == "new-rank"
    assert service.runtime()[2]._q is q


REAL_PROBE = endpoints._probe


@pytest.mark.parametrize("role,enabled", [
    ("embedding", True), ("visual_embedding", False),
    ("reranker", False), ("visual_reranker", False),
])
def test_form_save_checks_connection_before_persisting_even_when_off(setup, monkeypatch, role, enabled):
    from kb_search import rerank
    cfg, _ = setup
    payload = body(role, enabled=enabled, model_id="synthetic-model")
    response = Mock(ok=True)
    if "embedding" in role:
        count = min(payload["batch_size"], 2)
        response.json.return_value = {"data": [
            {"index": i, "embedding": [0.1] * payload["dim"]} for i in range(count)
        ]}
    else:
        response.json.return_value = {"results": [
            {"index": 0, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.2}
        ]}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    monkeypatch.setattr(rerank._session, "post", post)
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    saved = endpoints.save_endpoint(role, payload)
    post.assert_called_once()
    assert saved["enabled"] is enabled
    assert saved["model_id"] == "synthetic-model"
    if role == "embedding" or "reranker" in role:
        assert saved["validated"] is True
    before = read_overrides(cfg.state_db)
    post.side_effect = requests.ConnectionError("offline")
    with pytest.raises(ValueError, match="endpoint unavailable"):
        endpoints.save_endpoint(role, body(role, api_key="replacement-key"))
    assert read_overrides(cfg.state_db) == before


@pytest.mark.parametrize("role", ["visual_embedding", "reranker", "visual_reranker"])
def test_switch_only_can_disable_an_offline_endpoint(setup, monkeypatch, role):
    from kb_search import rerank
    post = Mock(side_effect=requests.ConnectionError("offline"))
    monkeypatch.setattr(endpoints.requests, "post", post)
    monkeypatch.setattr(rerank._session, "post", post)
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    saved = endpoints.save_endpoint(role, {"enabled": False, "revision": body(role)["revision"]})
    assert saved["enabled"] is False
    post.assert_not_called()


@pytest.mark.parametrize("role", ["visual_embedding", "reranker", "visual_reranker"])
def test_disabled_form_cannot_save_an_empty_endpoint(setup, role):
    cfg, _ = setup
    before = read_overrides(cfg.state_db)
    with pytest.raises(ValueError, match="API address is required"):
        endpoints.save_endpoint(role, body(role, enabled=False, base_url=""))
    assert read_overrides(cfg.state_db) == before


@pytest.mark.parametrize("role", ["embedding", "visual_embedding"])
def test_throughput_save_preserves_vector_space_and_is_allowed_during_graph_work(setup, role):
    cfg, env = setup
    original = endpoint_values(cfg)
    with db.connect(cfg.state_db) as con:
        con.execute("INSERT INTO graph_builds(graph_build_id,source_key,kb_id,source_collection,graph_version,status,started_at) "
                    "VALUES ('g','s','k','c','v','running',1)")
    endpoints.save_endpoint(role, body(role, batch_size=3, concurrency=4, api_key=""))
    fresh = config.load_settings(env)
    assert getattr(fresh, f"{role}_batch") == 3
    assert getattr(fresh, f"{role}_concurrency") == 4
    assert getattr(apply_overrides(cfg), f"{role}_concurrency") == 4
    values = endpoint_values(fresh)
    assert values[role] == original[role] | {"batch_size": 3, "concurrency": 4}
    assert all(values[r] == original[r] for r in original if r != role)


@pytest.mark.parametrize("role", ["reranker", "visual_reranker"])
def test_embedding_throughput_fields_do_not_leak_into_reranking(setup, role):
    with pytest.raises(ValueError, match="Unknown"):
        endpoints.save_endpoint(role, body(role, batch_size=2))
    assert "concurrency" not in body(role)


def test_visual_batch_probe_checks_two_independent_conversations(setup, monkeypatch):
    row = endpoint_values(setup[0])["visual_embedding"] | {"batch_size": 12}
    response = Mock()
    response.json.return_value = {"data": [{"index": i, "embedding": [0.1] * row["dim"]} for i in (1, 0)]}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    REAL_PROBE("visual_embedding", row)
    messages = post.call_args.kwargs["json"]["messages"]
    assert len(messages) == 2
    assert all(len(chat) == 2 and chat[1]["content"][0]["type"] == "image_url" for chat in messages)
    # A provider accepting the body but returning a fused vector is not batch capable.
    response.json.return_value["data"] = response.json.return_value["data"][:1]
    with pytest.raises(ValueError, match="batch size"):
        REAL_PROBE("visual_embedding", row)


@pytest.mark.parametrize("role,dim", [("embedding", 3), ("visual_embedding", 4)])
def test_embedding_probe_checks_real_request_shape_and_dimensions(setup, monkeypatch, role, dim):
    row = endpoint_values(setup[0])[role]
    count = min(row["batch_size"], 2)
    response = Mock(); response.json.return_value = {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(count)]}
    post = Mock(return_value=response); monkeypatch.setattr(endpoints.requests, "post", post)
    REAL_PROBE(role, row)
    request = post.call_args.kwargs
    assert request["json"]["dimensions"] == dim
    assert request["allow_redirects"] is False
    assert ("messages" in request["json"]) == (role == "visual_embedding")
    response.json.return_value = {"data": [{"index": i, "embedding": [0.1] * (dim + 1)} for i in range(count)]}
    with pytest.raises(ValueError, match="dimensions"):
        REAL_PROBE(role, row)


def test_text_embedding_validation_requires_success_and_matching_dimensions(setup, monkeypatch):
    cfg, _ = setup
    assert endpoints.get_endpoints()["endpoints"][0]["validated"] is False
    response = Mock()
    response.json.return_value = {"data": [
        {"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(2)
    ], "provider_trace": "do-not-store"}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    saved = endpoints.save_endpoint("embedding", body("embedding"))
    assert saved["validated"] is True and saved["enabled"] is True
    assert post.call_args.kwargs["json"]["dimensions"] == 3
    persisted = read_overrides(cfg.state_db)["embedding"]
    assert set(persisted) == {"base_url", "model_id", "api_key", "dim", "batch_size",
                              "concurrency", "enabled", "_validated_signature"}
    assert "_validated_signature" not in saved
    assert "do-not-store" not in json.dumps(persisted)
    post.reset_mock()
    assert endpoints.get_endpoints()["endpoints"][0]["validated"] is True
    post.assert_not_called()
    for field, value in (("base_url", "https://other.invalid/v1"), ("model_id", "other-model"),
                         ("api_key", "other-key"), ("dim", 4)):
        with db.connect(cfg.state_db) as con:
            db.set_app_config(con, CONFIG_KEY, {"embedding": {**persisted, field: value}})
        assert endpoints.get_endpoints()["endpoints"][0]["validated"] is False


@pytest.mark.parametrize("returned_dims", [(2, 2), (4, 4), (3, 2), (0, 0)])
def test_text_embedding_rejects_silent_dimension_changes_without_overwriting(setup, monkeypatch, returned_dims):
    cfg, _ = setup
    response = Mock()
    response.json.return_value = {"data": [
        {"index": i, "embedding": [0.1] * 3} for i in range(2)
    ]}
    monkeypatch.setattr(endpoints.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    endpoints.save_endpoint("embedding", body("embedding"))
    before = read_overrides(cfg.state_db)
    response.json.return_value = {"data": [
        {"index": i, "embedding": [0.1] * dim} for i, dim in enumerate(returned_dims)
    ]}
    with pytest.raises(ValueError, match="dimensions"):
        endpoints.save_endpoint("embedding", body("embedding", model_id="new-model"))
    assert read_overrides(cfg.state_db) == before


@pytest.mark.parametrize("result", [None, {}, {"data": []}, {"data": {}}, {"data": "bad"},
                                    {"data": [None, None]}, {"data": [{"index": 0}, {"index": 1}]}])
def test_text_embedding_rejects_missing_vectors_without_saving(setup, monkeypatch, result):
    response = Mock()
    response.json.return_value = result
    monkeypatch.setattr(endpoints.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    with pytest.raises(ValueError):
        endpoints.save_endpoint("embedding", body("embedding"))
    assert read_overrides(setup[0].state_db) == {}


def test_text_embedding_http_error_rejects_save_even_with_matching_vector_body(setup, monkeypatch):
    response = Mock(status_code=400)
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    response.json.return_value = {"data": [{"index": i, "embedding": [0.1] * 3} for i in range(2)]}
    monkeypatch.setattr(endpoints.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    with pytest.raises(ValueError, match="HTTP 400"):
        endpoints.save_endpoint("embedding", body("embedding"))
    assert read_overrides(setup[0].state_db) == {}


def test_failed_probe_does_not_echo_provider_secrets(setup, monkeypatch):
    response = Mock(status_code=401)
    error = requests.HTTPError("private-provider-body", response=response)
    monkeypatch.setattr(endpoints.requests, "post", Mock(side_effect=error))
    with pytest.raises(ValueError, match="HTTP 401") as caught:
        REAL_PROBE("embedding", endpoint_values(setup[0])["embedding"])
    assert "private-provider-body" not in str(caught.value)


@pytest.mark.parametrize("role", ["reranker", "visual_reranker"])
def test_reranker_probe_sends_real_request_and_accepts_nonempty_results(setup, monkeypatch, role):
    response = Mock(ok=True)
    response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8},
                                             {"index": 1, "relevance_score": 0.2}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    row = endpoint_values(setup[0])[role] | {"model_id": "synthetic-rank", "api_key": "test-key"}
    assert REAL_PROBE(role, row)["model_id"] == "synthetic-rank"
    request = post.call_args.kwargs
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "synthetic-rank"
    first = request["json"]["documents"][0]
    assert isinstance(first, dict) == (role == "visual_reranker")
    if role == "visual_reranker":
        from base64 import b64decode
        from io import BytesIO
        from PIL import Image
        assert request["json"]["query"]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        images = []
        for document in request["json"]["documents"]:
            assert [part["type"] for part in document["content"]] == ["image_url"]
            url = document["content"][0]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            with Image.open(BytesIO(b64decode(url.split(",", 1)[1]))) as image:
                assert image.size == (32, 32)
                images.append(image.tobytes())
        assert len(images) == 2 and images[0] != images[1]
    else:
        assert isinstance(request["json"]["query"], str)
    response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
    assert REAL_PROBE(role, row)["model_id"] == "synthetic-rank"


@pytest.mark.parametrize("role", ["reranker", "visual_reranker"])
@pytest.mark.parametrize("payload", [
    None, [], {"results": None}, {"results": {}}, {"results": []},
    {"error": "private-provider-body"},
])
def test_reranker_rejects_missing_results_without_saving(setup, monkeypatch, role, payload):
    response = Mock(ok=True)
    response.json.return_value = payload
    monkeypatch.setattr(endpoints.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    before = read_overrides(setup[0].state_db)
    with pytest.raises(ValueError, match="^Model connection test failed: invalid response$"):
        endpoints.save_endpoint(role, body(role, model_id="synthetic-rank", enabled=False))
    assert read_overrides(setup[0].state_db) == before


@pytest.mark.parametrize("role", ["reranker", "visual_reranker"])
@pytest.mark.parametrize("results", [
    [{"index": 1, "relevance_score": 0}, {"index": 0, "relevance_score": 0}],
    [{"index": 1, "relevance_score": 1}, {"index": 0, "relevance_score": 1}],
    [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
    [{"index": 0, "relevance_score": 0.8}],  # Fewer results is still a successful response.
    [{"index": 0, "relevance_score": 0.8}] * 2,
    [{"index": 99, "relevance_score": -10}],
    [{"relevance_score": "provider-specific-score"}],
    [{"index": 0}],
])
def test_reranker_connection_check_ignores_result_details_and_keeps_only_config_proof(setup, monkeypatch, role, results):
    cfg, _ = setup
    assert next(r for r in endpoints.get_endpoints()["endpoints"] if r["role"] == role)["validated"] is False
    response = Mock(ok=True)
    response.json.return_value = {"results": results, "provider_trace": "do-not-store"}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    saved = endpoints.save_endpoint(role, body(role, model_id="synthetic-rank"))
    assert saved["validated"] is True
    assert "_validated_signature" not in saved
    persisted = read_overrides(cfg.state_db)[role]
    assert set(persisted) == {"base_url", "model_id", "api_key", "enabled", "_validated_signature"}
    assert "do-not-store" not in json.dumps(read_overrides(cfg.state_db))
    post.reset_mock()
    off = endpoints.save_endpoint(role, {"enabled": False, "revision": saved["revision"]})
    assert off["validated"] is True and off["enabled"] is False
    assert next(r for r in endpoints.get_endpoints()["endpoints"] if r["role"] == role)["validated"] is True
    post.assert_not_called()  # Reading configuration and switching off do not probe.
    for field in ("base_url", "model_id", "api_key"):
        changed = {**persisted, field: "changed"}
        with db.connect(cfg.state_db) as con:
            db.set_app_config(con, CONFIG_KEY, {role: changed})
        current = next(r for r in endpoints.get_endpoints()["endpoints"] if r["role"] == role)
        assert current["validated"] is False


def test_explicit_reranker_probe_checks_disabled_endpoint_without_saving(setup, monkeypatch):
    response = Mock(ok=True)
    response.json.return_value = {"results": [{"index": 0, "relevance_score": 0},
                                             {"index": 1, "relevance_score": 0}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(endpoints.requests, "post", post)
    monkeypatch.setattr(endpoints, "_probe", REAL_PROBE)
    result = endpoints.probe_endpoint("reranker", body("reranker", enabled=False))
    assert result == {"ok": True, "model_id": "synthetic-rank"}
    post.assert_called_once()
    assert read_overrides(setup[0].state_db) == {}


def test_corrupt_configuration_fails_instead_of_silently_using_another_model(setup):
    cfg, env = setup
    with db.connect(cfg.state_db) as con:
        db.set_app_config(con, CONFIG_KEY, ["invalid"])
    with pytest.raises(ValueError, match="Invalid model endpoint configuration"):
        config.load_settings(env)


def test_visual_rerank_uses_saved_model_and_authentication(tmp_path, monkeypatch):
    from kb_search import rerank
    from PIL import Image
    import time
    image = tmp_path / "sample.png"; Image.new("RGB", (32, 32)).save(image)
    monkeypatch.setattr("kb_search.images._cached_file", lambda *a: image)
    # This contract test checks model/auth forwarding, not tokenizer cache startup.
    monkeypatch.setattr("kb_search.text.truncate_tokens", lambda text, limit: text)
    client = Mock(); client.score.return_value = [0.8]
    constructor = Mock(return_value=client); monkeypatch.setattr(rerank, "Reranker", constructor)
    cfg = SimpleNamespace(visual_reranker_base_url="http://rank.invalid/v1",
                          visual_reranker_model_id="visual-test", visual_reranker_api_key="test-key")
    result = rerank.rerank_images(cfg, "example", [{"point_id": "p", "payload": {"text": "example"}}],
                                 image_bytes=None, until=time.time() + 10)
    assert result["scores"] == {"p": 0.8}
    assert constructor.call_args.kwargs["model_id"] == "visual-test"
    assert constructor.call_args.kwargs["api_key"] == "test-key"


def test_api_roles_and_same_origin_protection(setup, monkeypatch):
    from fastapi.testclient import TestClient
    from kb_server.main import create_app
    monkeypatch.delenv("KB_WEB_TOKEN", raising=False)
    client = TestClient(create_app())
    assert client.get('/api/model-endpoints').status_code == 200
    assert client.put('/api/model-endpoints/unknown', json={}).status_code == 404
    response = client.put('/api/model-endpoints/reranker', json=body('reranker'),
                          headers={'Origin': 'https://other.invalid'})
    assert response.status_code == 403


@pytest.mark.parametrize("role", ["llm", "embedding", "visual_embedding", "reranker", "visual_reranker"])
def test_key_reveal_requires_explicit_authorized_request_and_is_not_cached(setup, monkeypatch, role):
    from fastapi.testclient import TestClient
    from kb_server import service
    from kb_server.main import create_app
    cfg, _ = setup
    monkeypatch.setattr(service, "settings", lambda: cfg)
    with db.connect(cfg.state_db) as con:
        db.upsert_llm(con, name="Synthetic model", base_url="https://model.invalid/v1",
                      model_id="synthetic-llm", api_key="synthetic-chat-key")
    monkeypatch.setenv("KB_WEB_TOKEN", "synthetic-console-token")
    client = TestClient(create_app())
    auth = {"Authorization": "Bearer synthetic-console-token"}
    if role == "llm":
        path, payload, expected = "/api/llms/Synthetic%20model/key", {}, "synthetic-chat-key"
        listing = client.get('/api/llms', headers=auth).json()
        assert all('api_key' not in x for x in listing)
    else:
        path = f"/api/model-endpoints/{role}/key"
        listing = client.get('/api/model-endpoints', headers=auth).json()['endpoints']
        assert all('api_key' not in x for x in listing)
        payload = {'revision': next(x['revision'] for x in listing if x['role'] == role)}
        expected = endpoint_values(cfg)[role]['api_key']
    assert client.post(path, json=payload).status_code == 401
    assert client.post(path, json=payload, headers=auth | {'Origin': 'https://other.invalid'}).status_code == 403
    assert client.get(path, headers=auth).status_code == 404  # Falls through to the static mount.
    response = client.post(path, json=payload, headers=auth | {'Origin': 'http://testserver'})
    assert response.status_code == 200
    assert response.json() == {'api_key': expected}
    assert response.headers['cache-control'] == 'no-store'
    assert read_overrides(cfg.state_db) == {}  # Revealing never changes configuration.


def test_reveal_rejects_stale_or_missing_model(setup, monkeypatch):
    from kb_server import service
    cfg, _ = setup
    monkeypatch.setattr(service, 'settings', lambda: cfg)
    with pytest.raises(KeyError):
        service.reveal_llm_key('missing')
    with pytest.raises(KeyError):
        endpoints.reveal_endpoint_key('missing', {})
    with pytest.raises(ValueError, match='refresh'):
        endpoints.reveal_endpoint_key('embedding', {'revision': 'stale'})
