"""Native, versioned Lance graph vectors. No database-client compatibility shim."""
import hashlib
import json
import math
import time

from .light_data import one_of


class LightVectors:
    def __init__(self, data):
        self.data = data
        with data._db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS graph_vectors(
                name TEXT PRIMARY KEY, table_name TEXT UNIQUE NOT NULL, dimensions TEXT NOT NULL,
                created_at REAL NOT NULL)""")

    def names(self):
        with self.data._db() as con:
            return {r[0] for r in con.execute("SELECT name FROM graph_vectors")}

    def _assert_mutable(self, name):
        from ..vector.qdrant import graph_collection_short_name, parse_graph_collection_name
        parsed = parse_graph_collection_name(name)
        if parsed is None:
            return
        with self.data._db() as con:
            if not con.execute("SELECT 1 FROM sqlite_master WHERE name='graph_metadata'").fetchone():
                return
            rows = con.execute("SELECT collection FROM graph_metadata WHERE version=? AND sealed=1",
                               (parsed['graph_version'],)).fetchall()
        if any(graph_collection_short_name(col) == parsed['source_short'] for col, in rows):
            raise ValueError('published graph versions are immutable')

    def _entry(self, name):
        with self.data._db() as con:
            row = con.execute("SELECT table_name,dimensions FROM graph_vectors WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        return row

    def table(self, name):
        return self.data.connection.open_table(self._entry(name)[0])

    def ensure(self, name, size, layout=None):
        import pyarrow as pa
        dims = {key: int(size) for key in (layout or ["vector"])}
        if size <= 0:
            raise ValueError("vector dimension must be positive")
        with self.data._write_lock(), self.data._db() as con:
            previous = con.execute("SELECT dimensions FROM graph_vectors WHERE name=?", (name,)).fetchone()
            if previous:
                if json.loads(previous[0]) != dims:
                    raise ValueError("graph vector layout mismatch")
                return "exists"
            self._assert_mutable(name)
            table_name = "vectors_" + hashlib.sha256(name.encode()).hexdigest()
            schema = pa.schema([("id", pa.string()), ("payload", pa.string()),
                                *((key, pa.list_(pa.float32(), n)) for key, n in dims.items())])
            table = self.data.connection.create_table(table_name, schema=schema, exist_ok=True)
            if table.schema != schema:
                raise ValueError("unpublished graph vector layout mismatch")
            con.execute("INSERT INTO graph_vectors VALUES(?,?,?,?)", (name, table_name, json.dumps(dims), time.time()))
        return "created"

    def write(self, name, ids, vectors, payloads):
        import pyarrow as pa
        dims = json.loads(self._entry(name)[1])
        rows = []
        for ident, value, payload in zip(ids, vectors, payloads, strict=True):
            value = value if isinstance(value, dict) else {"vector": value}
            if set(value) != set(dims):
                raise ValueError("graph vector names mismatch")
            row = dict(id=str(ident), payload=json.dumps(payload, ensure_ascii=False, allow_nan=False))
            for key, size in dims.items():
                vector = value[key]
                if len(vector) != size or any(not math.isfinite(float(x)) or abs(float(x)) > 3.4028234663852886e38 for x in vector):
                    raise ValueError("invalid graph vector")
                row[key] = list(map(float, vector))
            rows.append(row)
        if rows:
            with self.data._write_lock():
                self._assert_mutable(name)
                table = self.table(name)
                table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                    pa.Table.from_pylist(rows, schema=table.schema))

    def rows(self, name, ids=None, *, vectors=False):
        table = self.table(name)
        query = table.search().limit(None)
        if ids is not None:
            query = query.where(one_of("id", ids))
        if not vectors:
            query = query.select(["id", "payload"])
        return [{**r, "payload": json.loads(r["payload"])} for r in query.to_list()]

    def lookup(self, name, ids):
        rows = self.rows(name, ids, vectors=True)
        dims = json.loads(self._entry(name)[1])
        return {r["id"]: (r["payload"], r["vector"] if list(dims) == ["vector"] else {k: r[k] for k in dims}) for r in rows}

    def search(self, name, vector, limit, *, using=None, predicate=None):
        if name not in self.names() or limit <= 0:
            return []
        table = self.table(name)
        column = using or "vector"
        if column not in table.schema.names:
            raise ValueError("unknown graph vector name")
        if len(vector) != table.schema.field(column).type.list_size:
            raise ValueError("graph query vector dimension mismatch")
        # Source-time and quality predicates apply before top-k, including on
        # sparse historical slices. Exact search avoids ANN recall loss.
        ids = [r["id"] for r in self.rows(name) if predicate is None or predicate(r["payload"])]
        if not ids:
            return []
        rows = (table.search(vector, vector_column_name=column).distance_type("cosine").bypass_vector_index()
                .where(one_of("id", ids), prefilter=True).limit(limit).select(["id", "payload", "_distance"]).to_list())
        return [dict(json.loads(r["payload"]), _point_id=r["id"], _score=1 - float(r["_distance"])) for r in rows]

    def count(self, name):
        return self.table(name).count_rows()

    def delete_except(self, name, ids):
        with self.data._write_lock():
            self._assert_mutable(name)
            table = self.table(name)
            where = "NOT (" + one_of("id", ids) + ")"
            count = table.count_rows(where)
            if count:
                table.delete(where)
        return count

    def drop(self, name):
        with self.data._write_lock(), self.data._db() as con:
            row = con.execute("SELECT table_name FROM graph_vectors WHERE name=?", (name,)).fetchone()
            if row:
                self.data.connection.drop_table(row[0], ignore_missing=True)
                con.execute("DELETE FROM graph_vectors WHERE name=?", (name,))
