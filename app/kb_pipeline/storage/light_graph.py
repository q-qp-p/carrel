"""Light graph projection and atomic version publication.

LanceDB is the persistent graph store. DuckDB executes bounded, in-process joins
over immutable published tables. This storage primitive is not a Cypher adapter;
the application adapter exposes domain operations without translating Cypher.
"""
from contextlib import contextmanager
import hashlib
from itertools import chain
import json
from tempfile import TemporaryDirectory
import threading
import time

from .light_data import LightChunks


NODE_KINDS = {"documents": "Document", "text_units": "TextUnit", "qdrant_chunks": "QdrantChunkSnapshot",
              "entities": "Entity", "relations": "Relation", "specs": "Spec"}
EDGE_KINDS = {
    "document_edges": ("HAS_TEXT_UNIT", "document_uid", "text_unit_uid"),
    "contributes_to_edges": ("CONTRIBUTES_TO", "chunk_uid", "text_unit_uid"),
    "mentions_edges": ("MENTIONS", "text_unit_uid", "target_uid"),
    "evidences_edges": ("EVIDENCES", "text_unit_uid", "target_uid"),
    "mentioned_in_edges": ("MENTIONED_IN", "entity_uid", "chunk_uid"),
    "has_spec_edges": ("HAS_SPEC", "entity_uid", "spec_uid"),
    "of_property_edges": ("OF_PROPERTY", "spec_uid", "entity_uid"),
    "states_edges": ("STATES", "text_unit_uid", "spec_uid"),
}


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LightGraphProjection:
    def __init__(self, data: LightChunks):
        self.data = data
        with data._db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS graph_versions(
                kb_id TEXT NOT NULL, version TEXT NOT NULL, generation TEXT NOT NULL,
                nodes INTEGER NOT NULL, edges INTEGER NOT NULL, PRIMARY KEY(kb_id,version))""")
            con.execute("CREATE TABLE IF NOT EXISTS graph_active(kb_id TEXT PRIMARY KEY, version TEXT NOT NULL)")

    def active_version(self, kb_id):
        with self.data._db() as con:
            row = con.execute("SELECT version FROM graph_active WHERE kb_id=?", (kb_id,)).fetchone()
        return row[0] if row else None

    def _version(self, kb_id, version=None):
        with self.data._db() as con:
            if version is None:
                row = con.execute("""SELECT v.version,v.generation,v.nodes,v.edges FROM graph_versions v
                    JOIN graph_active a ON a.kb_id=v.kb_id AND a.version=v.version WHERE v.kb_id=?""", (kb_id,)).fetchone()
            else:
                row = con.execute("SELECT version,generation,nodes,edges FROM graph_versions WHERE kb_id=? AND version=?",
                                  (kb_id, version)).fetchone()
        if row is None:
            raise KeyError(f"graph version not found: {kb_id}")
        return row

    @staticmethod
    def _records(kb_id, version, bundle):
        nodes, edges = [], []
        for field, kind in NODE_KINDS.items():
            for row in bundle.get(field) or []:
                if row.get("kb_id") != kb_id or row.get("graph_version") != version:
                    raise ValueError("graph node belongs to a different library or version")
                nodes.append(dict(uid=str(row["uid"]), kind=kind, id=str(row.get("id") or row.get("point_id") or ""),
                                  doc_id=row.get("doc_id"), payload=compact(row)))
        for field, (kind, source, target) in EDGE_KINDS.items():
            for row in bundle.get(field) or []:
                edges.append(dict(kind=kind, src=row[source], dst=row[target], rel=None, predicate=None, payload=compact(row)))
        for row in bundle.get("relation_entity_edges") or []:
            relation = row["relation_uid"]
            props = row["related_props"]
            edges.append(dict(kind="RELATED_TO", src=row["source_uid"], dst=row["target_uid"], rel=relation,
                              predicate=props.get("type"), payload=compact(props)))
            edges.extend(dict(kind=kind, src=relation, dst=row[field], rel=None, predicate=None, payload="{}")
                         for kind, field in (("SOURCE", "source_uid"), ("TARGET", "target_uid")))
        ids = {row["uid"] for row in nodes}
        if len(ids) != len(nodes):
            raise ValueError("duplicate graph node IDs")
        if any(row["src"] not in ids or row["dst"] not in ids or (row["rel"] and row["rel"] not in ids) for row in edges):
            raise ValueError("graph edge has a missing endpoint")
        # Stable content-derived table names make retrying an interrupted stage
        # idempotent, without overwriting a table that a reader is using.
        nodes.sort(key=lambda row: row["uid"])
        edges.sort(key=compact)
        return nodes, edges

    def stage(self, kb_id, version, bundle):
        import pyarrow as pa
        nodes, edges = self._records(kb_id, version, bundle)
        digest = hashlib.sha256()
        for row in chain((kb_id, version), nodes, edges):
            digest.update(compact(row).encode())
            digest.update(b"\n")
        generation = "graph_" + digest.hexdigest()
        node_schema = pa.schema([(key, pa.string()) for key in ("uid", "kind", "id", "doc_id", "payload")])
        edge_schema = pa.schema([(key, pa.string()) for key in ("kind", "src", "dst", "rel", "predicate", "payload")])
        with self.data._write_lock():
            with self.data._db() as con:
                current = con.execute("SELECT generation FROM graph_versions WHERE kb_id=? AND version=?",
                                      (kb_id, version)).fetchone()
            if current and current[0] != generation:
                raise ValueError("cannot overwrite a staged graph version")
            for suffix, rows, schema in (("nodes", nodes, node_schema), ("edges", edges, edge_schema)):
                table = self.data.connection.create_table(generation + "_" + suffix,
                    data=pa.Table.from_pylist(rows, schema=schema), exist_ok=True)
                if table.count_rows() != len(rows):
                    raise RuntimeError("incomplete graph table")
            with self.data._db() as con:
                con.execute("INSERT OR REPLACE INTO graph_versions VALUES(?,?,?,?,?)",
                            (kb_id, version, generation, len(nodes), len(edges)))
        return dict(graph_version=version, generation=generation, nodes=len(nodes), edges=len(edges))

    def activate(self, kb_id, version, *, seal=False, validate=None):
        with self.data._write_lock():
            if validate is not None:
                validate()
            _, generation, nodes, edges = self._version(kb_id, version)
            for suffix, expected in (("nodes", nodes), ("edges", edges)):
                if self.data.connection.open_table(generation + "_" + suffix).count_rows() != expected:
                    raise RuntimeError("incomplete graph version cannot be published")
            with self.data._db() as con:
                if seal:
                    con.execute("UPDATE graph_metadata SET sealed=1 WHERE kb_id=? AND version=?", (kb_id, version))
                con.execute("INSERT INTO graph_active VALUES(?,?) ON CONFLICT(kb_id) DO UPDATE SET version=excluded.version",
                            (kb_id, version))
        return {"active_graph_version": version}

    def deactivate(self, kb_id):
        with self.data._write_lock(), self.data._db() as con:
            con.execute("DELETE FROM graph_active WHERE kb_id=?", (kb_id,))

    @contextmanager
    def session(self, kb_id, version=None, *, deadline=None):
        import duckdb
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Light graph query budget exhausted')
        selected, generation, _, _ = self._version(kb_id, version)
        # A reader pins an immutable generation before any database access;
        # switching the publication pointer cannot mix two graph versions.
        with TemporaryDirectory(prefix="duckdb-", dir=self.data.root) as scratch:
            con = duckdb.connect(":memory:", config={"threads": "2", "memory_limit": "512MB",
                "temp_directory": scratch, "max_temp_directory_size": "1GB", "preserve_insertion_order": "false"})
            timer = threading.Timer(max(0, deadline-time.monotonic()), con.interrupt) if deadline is not None else None
            if timer:
                timer.daemon = True
                timer.start()
            try:
                for name in ("nodes", "edges"):
                    table = self.data.connection.open_table(generation + "_" + name)
                    con.register("input_rows", table.search().limit(None).to_batches(batch_size=2048))
                    con.execute(f"CREATE TEMP TABLE {name} AS SELECT * FROM input_rows")
                    con.unregister("input_rows")
                yield selected, con
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Light graph query budget exhausted')
            finally:
                if timer:
                    timer.cancel()
                    timer.join()
                con.close()

    def nodes(self, kb_id, kind, *, version=None):
        with self.session(kb_id, version) as (_, con):
            return [json.loads(row[0]) for row in con.execute("SELECT payload FROM nodes WHERE kind=? ORDER BY uid", [kind]).fetchall()]

    def neighbors(self, kb_id, uid, *, version=None):
        with self.session(kb_id, version) as (_, con):
            return con.execute("""SELECT rel, CASE WHEN src=? THEN dst ELSE src END, src=?, predicate
                FROM edges WHERE kind='RELATED_TO' AND (src=? OR dst=?) ORDER BY rel""", [uid] * 4).fetchall()

    def evidence(self, kb_id, *, kind="EVIDENCES", uid=None, version=None):
        if kind not in ("EVIDENCES", "STATES"):
            raise ValueError("unknown evidence edge kind")
        with self.session(kb_id, version) as (_, con):
            return con.execute("""SELECT DISTINCT e.dst,n.id,n.doc_id FROM edges e
                JOIN edges c ON c.dst=e.src AND c.kind='CONTRIBUTES_TO'
                JOIN nodes n ON n.uid=c.src AND n.kind='QdrantChunkSnapshot'
                WHERE e.kind=?""" + (" AND e.dst=?" if uid is not None else "") + " ORDER BY e.dst,n.id",
                [kind, uid] if uid is not None else [kind]).fetchall()

    def facts(self, kb_id, *, history=False, as_of=None, conflict_only=False, version=None):
        from ..graph.currency import query_scope, in_scope
        scope = query_scope(history=history, as_of=as_of)
        return [row for row in self.nodes(kb_id, "Spec", version=version)
                if in_scope(row, scope) and (not conflict_only or row.get("conflict_group"))]
