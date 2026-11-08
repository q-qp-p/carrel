"""Native reclamation. Active versions and in-flight readers are never removed."""
import json
import re
import time
from ..vector.qdrant import graph_collection_short_name, parse_graph_collection_name
from .light_data import one_of, literal


def compact_chunks(data, *, dry_run=False):
    """Reclaim obsolete Lance files without expiring application graph history.

    Recent table snapshots and unverified writes retain Lance's seven-day grace
    period. This runs in the existing weekly job, not on the ingest/query path.
    """
    from datetime import timedelta
    optimized = []
    orphans = []
    with data._write_lock(), data._db() as con:
        for collection, table_name in con.execute('SELECT name,table_name FROM collections ORDER BY name'):
            if not dry_run:
                data.connection.open_table(table_name).optimize(
                    cleanup_older_than=timedelta(days=7), delete_unverified=False)
            optimized.append(collection)
        known = {r[0] for r in con.execute('SELECT table_name FROM collections')}
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'graph_vectors' in tables:
            known.update(r[0] for r in con.execute('SELECT table_name FROM graph_vectors'))
        if 'graph_versions' in tables:
            known.update(r[0]+'_'+suffix for r in con.execute('SELECT generation FROM graph_versions')
                         for suffix in ('nodes','edges'))
        # A process can stop after creating a native table but before registering
        # it. Only Carrel-owned names without any manifest reference are eligible.
        for path in sorted((data.root/'lance').glob('*.lance')):
            name = path.name.removesuffix('.lance')
            if (name in known or not re.fullmatch(r'(?:chunks_|vectors_|graph_)[0-9a-f]{64}(?:_nodes|_edges)?',name)
                    or path.stat().st_mtime >= time.time()-7*86400):
                continue
            if not dry_run:
                data.connection.drop_table(name,ignore_missing=True)
            orphans.append(name)
    return dict(dry_run=dry_run, collections=optimized, orphan_tables=orphans, snapshot_retention_days=7)


def _remove_version(graph, kb, version, collection):
    data = graph.store.chunks
    # Callers hold the exclusive reader lock. Remove the manifest first; a killed
    # cleanup leaves unreferenced tables, never a published pointer to missing data.
    with data._write_lock(), data._db() as con:
        row = con.execute('SELECT generation FROM graph_versions WHERE kb_id=? AND version=?',(kb,version)).fetchone()
        con.execute('DELETE FROM graph_versions WHERE kb_id=? AND version=?',(kb,version))
        con.execute('DELETE FROM graph_metadata WHERE kb_id=? AND version=?',(kb,version))
    if row:
        with data._write_lock():
            for suffix in ('nodes','edges'):
                data.connection.drop_table(row[0]+'_'+suffix,ignore_missing=True)
    for name in list(graph.vectors.names()):
        parsed = parse_graph_collection_name(name)
        if parsed and parsed.get('source_short') == graph_collection_short_name(collection) and parsed.get('graph_version',parsed.get('version')) == version:
            graph.vectors.drop(name)


def collect_graph(graph, *, source_collections, retention_days, dry_run=False, delete_unparseable=False,
                  keep_latest=None, discard_versions=(), protect_versions=(), version_timestamps=None):
    if retention_days < 1 or (keep_latest is not None and keep_latest < 1):
        raise ValueError('retention and keep count must be positive')
    data = graph.store.chunks
    deleted, skipped = [], []
    with graph._read_lock(exclusive=True):
        with data._db() as con:
            rows = con.execute('SELECT kb_id,version,collection,created_at FROM graph_metadata ORDER BY created_at DESC,version DESC').fetchall()
            active = dict(con.execute('SELECT kb_id,version FROM graph_active'))
        groups = {}
        for kb,version,col,created in rows:
            if col in source_collections:
                groups.setdefault(col,[]).append((kb,version,created))
        for col, entries in groups.items():
            slots = max(0,(keep_latest or 0)-int(any(active.get(kb)==v for kb,v,_ in entries)))
            for kb,version,created in entries:
                reason = None
                if active.get(kb) == version: reason='active_version'
                elif version in protect_versions: reason='resumable'
                elif version in discard_versions: pass
                elif keep_latest is not None and slots:
                    reason='within_keep_latest'; slots-=1
                elif keep_latest is None and created >= time.time()-retention_days*86400: reason='within_retention'
                row=dict(kb_id=kb,graph_version=version,source_collection=col)
                if reason:
                    skipped.append(dict(row,reason=reason))
                else:
                    if not dry_run: _remove_version(graph,kb,version,col)
                    deleted.append(dict(row,reason='expired',status='dry_run' if dry_run else 'deleted'))
        # Failed vector writes can precede the projection manifest. Age-based
        # collection handles these; resumable versions are kept explicitly.
        known={(col,v) for _,v,col,_ in rows}
        for name in sorted(graph.vectors.names()):
            parsed=parse_graph_collection_name(name)
            if not parsed: continue
            col=next((c for c in source_collections if graph_collection_short_name(c)==parsed['source_short']),None)
            version=parsed['graph_version']
            if not col or (col,version) in known: continue
            if version in protect_versions:
                skipped.append(dict(collection=name,graph_version=version,reason='resumable'));continue
            with data._db() as con:
                created=con.execute('SELECT created_at FROM graph_vectors WHERE name=?',(name,)).fetchone()[0]
            if version not in discard_versions and created >= time.time()-retention_days*86400:
                skipped.append(dict(collection=name,graph_version=version,reason='within_retention'));continue
            if not dry_run: graph.vectors.drop(name)
            deleted.append(dict(collection=name,graph_version=version,reason='orphan'))
        deleted.extend(_orphan_projections(graph, source_collections, retention_days, protect_versions, dry_run))
    return dict(dry_run=dry_run,deleted=deleted,skipped_collections=skipped,errors=[])


def _orphan_projections(graph, collections, retention_days, protected, dry_run):
    """Reclaim interrupted stages only after retention, under the writer/reader locks."""
    data = graph.store.chunks
    removed = []
    with data._write_lock(), data._db() as con:
        versions = {r[2]: (r[0], r[1]) for r in con.execute('SELECT kb_id,version,generation FROM graph_versions')}
        known = set(con.execute('SELECT kb_id,version FROM graph_metadata'))
        known.update(con.execute('SELECT kb_id,version FROM graph_active'))
        sources = {s.kb_id for s in graph.settings.sources.values() if s.collection in collections}
        for path in (data.root / 'lance').glob('graph_*_nodes.lance'):
            generation = path.name.removesuffix('_nodes.lance')
            identity = versions.get(generation)
            if identity in known:
                continue
            if identity is None:
                rows = data.connection.open_table(generation+'_nodes').search().limit(1).select(['payload']).to_list()
                if not rows:
                    continue  # An empty unknown stage has no source scope to authorize deletion.
                payload = json.loads(rows[0]['payload'])
                identity = (payload['kb_id'], payload['graph_version'])
            if identity[0] not in sources or identity[1] in protected or path.stat().st_mtime >= time.time()-retention_days*86400:
                continue
            if not dry_run:
                con.execute('DELETE FROM graph_versions WHERE kb_id=? AND version=?', identity)
                # No active/complete metadata references these tables.
                for suffix in ('nodes','edges'):
                    data.connection.drop_table(generation+'_'+suffix, ignore_missing=True)
            removed.append(dict(kb_id=identity[0],graph_version=identity[1],reason='orphan_projection',status='dry_run' if dry_run else 'deleted'))
    return removed


def drop_graph(graph,kb_id,collection):
    with graph._read_lock(exclusive=True):
        graph.projection.deactivate(kb_id)
        with graph.store.chunks._db() as con:
            versions=[r[0] for r in con.execute('SELECT version FROM graph_versions WHERE kb_id=?',(kb_id,))]
        for version in versions:
            _remove_version(graph,kb_id,version,collection)
        for name in list(graph.vectors.names()):
            parsed=parse_graph_collection_name(name)
            if parsed and parsed['source_short']==graph_collection_short_name(collection): graph.vectors.drop(name)
    return dict(versions_deleted=versions)


def drop_collection(data,collection):
    with data._write_lock(), data._db() as con:
        row=con.execute('SELECT table_name FROM collections WHERE name=?',(collection,)).fetchone()
        if row:
            data.connection.drop_table(row[0],ignore_missing=True)
            con.execute('DELETE FROM collections WHERE name=?',(collection,))
    return {'collection':collection,'deleted':bool(row)}


def inactive_gc(data,collection,cutoff,*,dry_run=False):
    if not data.exists(collection): return 0
    with data._write_lock():
        rows=data._rows(collection,'is_active = false')
        ids=[r['id'] for r in rows if (p:=json.loads(r['payload'])).get('inactive_at') is not None and p['inactive_at'] < cutoff]
        if ids and not dry_run: data._table(collection).delete(one_of('id',ids))
    return len(ids)


def versions(data,collection,doc_id):
    from ..maintenance import _safe_version
    if not data.exists(collection): return set()
    return {_safe_version(str(json.loads(r['payload']).get('content_version') or ''))
            for r in data._rows(collection,'doc_id = '+literal(doc_id))}


def _expired(data,collection,cutoff):
    if not data.exists(collection): return []
    return [(r['id'],p) for r in data._rows(collection,'is_active = false')
            if (p:=json.loads(r['payload'])).get('inactive_at') is not None and p['inactive_at'] < cutoff]


def inactive_doc_versions_older_than(data,collection,cutoff_ts):
    from collections import Counter
    groups=Counter((p.get('doc_id'),p.get('content_version')) for _,p in _expired(data,collection,cutoff_ts)
                   if p.get('doc_id') and p.get('content_version'))
    return [dict(collection=collection,doc_id=k[0],content_version=k[1],points=n) for k,n in sorted(groups.items())]


def expired_inactive_point_ids(data,collection,*,doc_id,content_version,cutoff_ts):
    return [i for i,p in _expired(data,collection,cutoff_ts) if p.get('doc_id')==doc_id and p.get('content_version')==content_version]


def delete_expired_inactive_points(data,collection,point_ids,cutoff_ts):
    with data._write_lock():
        wanted=set(point_ids)
        ids=[i for i,_ in _expired(data,collection,cutoff_ts) if i in wanted]
        if ids: data._table(collection).delete(one_of('id',ids))
    return len(ids)


def delete_malformed_inactive_points(data,collection,cutoff_ts,*,dry_run=False):
    with data._write_lock():
        ids=[i for i,p in _expired(data,collection,cutoff_ts) if not p.get('doc_id') or not p.get('content_version')]
        if ids and not dry_run: data._table(collection).delete(one_of('id',ids))
    return len(ids)


def active_doc_point_count(data,collection,*,kb_id,file_key):
    if not data.exists(collection): return 0
    return data._table(collection).count_rows('is_active = true AND doc_id = '+literal(f'{kb_id}:{file_key}'))


def backfill_inactive_at(data,collection,inactive_at_ts,*,dry_run=False):
    if not data.exists(collection): return 0
    ids=[r['id'] for r in data._rows(collection,'is_active = false') if json.loads(r['payload']).get('inactive_at') is None]
    if ids and not dry_run:
        data._mutate(collection,one_of('id',ids),lambda p:p.update(inactive_at=inactive_at_ts))
    return len(ids)
