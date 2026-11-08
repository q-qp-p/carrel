"""Light graph-neighborhood lookup using shared subject selection and output rules."""
from contextlib import ExitStack
from collections import defaultdict
import time
from typing import Any
from kb_search.graphwalk import (_name_key, named_entities, time_scope, NAME_MIN_CHARS, NAME_MAX_CHARS,
    NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SCAN, NEIGHBORHOOD_KINDS)
from .light_reader import live


def neighborhoods(graph, sources, graph_versions, question, seeds, *, limit, per_entity=NEIGHBORHOOD_SIZE,
                  timeout=None, scope=None, hit_points=None):
    from kb_search import docstatus
    scope = scope or time_scope()
    deadline = time.monotonic() + timeout if timeout is not None else None
    graph_versions = {kb: gv for kb,gv in graph_versions.items() if gv and kb in sources and
        (scope['mode'] != 'current' or not docstatus.graph_refresh_pending(graph.settings,sources[kb]))}
    with ExitStack() as stack:
        readers = {kb:stack.enter_context(graph.reader(sources[kb],gv,
            timeout=max(0,deadline-time.monotonic()) if deadline is not None else None))
            for kb,gv in graph_versions.items()}
        indexes = {}
        for kb, reader in readers.items():
            names, entities = {}, {}
            for r in reader.nodes('Entity'):
                if not live(r):
                    continue
                eid = r['id']
                entities[eid] = {**r, 'degree': int(r.get('degree') or 0)}
                for i, name in enumerate([r.get('title'), *(r.get('aliases') or [])]):
                    key = _name_key(name)
                    if NAME_MIN_CHARS <= len(key) <= NAME_MAX_CHARS:
                        names.setdefault(key, []).append((eid,i == 0))
            indexes[kb] = dict(names=names, entities=entities, gv=reader.version)
        if hit_points is not None:
            counts = defaultdict(lambda: [set(),0])
            for kb, reader in readers.items():
                wanted = {pid for k,pid in hit_points if k == kb}
                for r in reader.mentions(list(indexes[kb]['entities'])):
                    if r['point_id'] in wanted:
                        counts[kb,r['eid']][0].add(r['point_id'])
                        counts[kb,r['eid']][1] += r.get('count') or 1
            seeds = [dict(kb_id=k[0],id=k[1]) for k,v in sorted(counts.items(),key=lambda x:(-len(x[1][0]),-x[1][1],x[0]))[:4]]
        seeded = {(str(e.get("kb_id")), str(e.get("id"))) for e in seeds}
        named = [{**e, "kb_id": kb, "named": True} for kb, index in indexes.items() for e in named_entities(question, index)
                 if e["_rank"] >= 2 or (kb, e["id"]) in seeded]
        order = {kb: i for i, kb in enumerate(graph_versions)}
        named.sort(key=lambda e: (-e["_rank"], order[e["kb_id"]], -e["degree"]))
        subjects = named[:limit]
        taken = {(s["kb_id"], s["id"]) for s in subjects}
        used = {(s["kb_id"], s["_key"]) for s in subjects}
        for seed in seeds:
            if len(subjects) >= limit:
                break
            kb, eid = str(seed.get("kb_id")), str(seed.get("id") or "")
            entity = (indexes.get(kb) or {"entities": {}})["entities"].get(eid)
            if entity is None or (kb, eid) in taken or (kb, _name_key(entity.get("title"))) in used:
                continue
            taken.add((kb, eid))
            used.add((kb, _name_key(entity.get("title"))))
            subjects.append({**entity, "kb_id": kb, "named": False})
        rel_rows, facts = {}, {}
        for kb, reader in readers.items():
            ids = [s['id'] for s in subjects if s['kb_id'] == kb and not s.get('ambiguous')]
            for eid in ids:
                found = reader.adjacent([eid],scope,traversable=True)
                rel_rows[kb,eid] = dict(total=len(found),kinds=[r.get('type') for _,r,_,_ in found],top=[
                    dict(relation_id=r['id'],type=r.get('type'),outgoing=outgoing,directed=r.get('directed',False),weight=r.get('weight'),
                         other_id=n['id'],other_title=n.get('title'),other_type=n.get('type'),other_degree=n.get('degree'))
                    for _,r,n,outgoing in found[:per_entity * NEIGHBORHOOD_SCAN]])
                facts[kb,eid] = reader.fact_count(eid,scope)
    out: list[dict[str, Any]] = []
    for s in subjects:
        if s.get("ambiguous"):
            out.append({"kb_id": s["kb_id"], "id": s["id"], "title": s.get("title"), "type": s.get("type"),
                        "scope": s.get("scope"), "graph_version": graph_versions[s["kb_id"]],
                        "named": True, "ambiguous": True, "neighbors": []})
            continue
        key = (s["kb_id"], s["id"])
        row = rel_rows.get(key) or {}
        counts: dict[str, int] = {}
        for kind in row.get("kinds") or []:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
        total = int(row.get("total") or 0)
        if not s["named"] and not total and not facts.get(key):
            continue
        top: dict[str, dict[str, Any]] = {}
        for t in row.get("top") or []:
            if len(top) < per_entity:
                top.setdefault(str(t.get("other_id")), t)
        out.append({"kb_id": s["kb_id"], "id": s["id"], "title": s.get("title"), "type": s.get("type"), "named": bool(s["named"]),
                    "scope": s.get("scope"), "graph_version": graph_versions[s["kb_id"]],
                    "relations": total, "facts": facts.get(key, 0),
                    "predicates": [{"type": k, "count": n} for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:NEIGHBORHOOD_KINDS]],
                    "neighbors": [{"relation_id": t.get("relation_id"), "type": t.get("type"), "direction": "out" if t.get("outgoing") else "in",
                                   "directed": bool(t.get("directed")), "weight": t.get("weight"),
                                   "other": {"id": t.get("other_id"), "title": t.get("other_title"), "type": t.get("other_type"),
                                             "degree": t.get("other_degree")}} for t in top.values()]})
    return out
