"""Manual start readiness reuses pipeline rules without running model or indexing services."""
from types import SimpleNamespace
from unittest import mock

import pytest

from kb_server import service
from kb_pipeline.graph import build


@pytest.mark.parametrize('append,rebuild,kind', [
    ({'reason':'no_successful_build'}, {'due':False}, 'build'),
    ({'due':True}, {'due':False}, 'append'),
    ({'due':True}, {'due':True}, 'build'),
    ({'reason':'source_unchanged'}, {'due':True}, 'build'),
    ({'reason':'source_unchanged'}, {'due':False}, 'none'),
    ({'reason':'no_active_content'}, {'due':True}, 'none'),
    ({'reason':'paused_by_operator'}, {'due':True}, 'none'),
    ({'reason':'config_changed_needs_full_rebuild'}, {'due':False}, 'none'),
    ({'reason':'llm_not_configured'}, {'due':True}, 'none'),
])
def test_manual_graph_readiness(append, rebuild, kind):
    cfg, source = object(), object()
    with mock.patch.object(build, 'evaluate_append', return_value=append) as check:
        result = service._graph_run_decision(cfg, 'synthetic', source, rebuild=rebuild)
    assert result['kind'] == kind
    check.assert_called_once_with(cfg, source_key='synthetic', source=source, ignore_auto_flag=True)


def test_manual_readiness_cache_refreshes_when_work_changes():
    cfg = SimpleNamespace(sources={'synthetic':SimpleNamespace(graph_enabled=True)})
    service._rebuild_check_cache.clear()
    try:
        with mock.patch.object(build, 'evaluate_rebuild', return_value={'due':False}), \
                mock.patch.object(build, 'evaluate_append', side_effect=[{'reason':'source_unchanged'}, {'due':True}]) as append:
            first = service.rebuild_check_cached(cfg, 'synthetic', now=100, revision=('indexed-v1',))
            same = service.rebuild_check_cached(cfg, 'synthetic', now=101, revision=('indexed-v1',))
            changed = service.rebuild_check_cached(cfg, 'synthetic', now=102, revision=('indexed-v2',))
        assert first is same
        assert first['manual_run']['kind'] == 'none'
        assert changed['manual_run']['kind'] == 'append'
        assert append.call_count == 2
    finally:
        service._rebuild_check_cache.clear()
