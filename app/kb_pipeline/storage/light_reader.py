"""Graph-domain reads over a pinned Lance generation, joined by DuckDB.

Scoring, temporal interpretation and evidence formatting stay in the shared query
modules. This reader exposes operations, not a Cypher or database-client facade.
"""
from collections import Counter, defaultdict
import json

from ..graph.currency import in_scope
from ..graph import recall
from ..vector.qdrant import graph_collection_name


def live(row):
    return not row.get("boilerplate") and not row.get("reference")


class LightReader:
    def __init__(self, store, source, version, con):
        self.store, self.source, self.version, self.con = store, source, version, con
        self._nodes = {}

    def nodes(self, kind):
        if kind not in self._nodes:
            self._nodes[kind] = [json.loads(r[0]) for r in self.con.execute(
                "SELECT payload FROM nodes WHERE kind=? ORDER BY id", [kind]).fetchall()]
        return self._nodes[kind]

    def collection(self, kind):
        return graph_collection_name(self.source.collection, kind, self.version)

    def dense(self, kind, vector, limit, scope, using=None):
        return self.store.graph.vectors.search(self.collection(kind), vector, limit, using=using,
            predicate=(None if kind == 'page' else lambda r: live(r) and (kind == 'entity' or in_scope(r, scope))))

    def payloads(self, kind):
        name = self.collection(kind)
        if name not in self.store.graph.vectors.names():
            return []
        return [{**r['payload'], '_point_id': r['id']} for r in self.store.graph.vectors.rows(name)]

    def lexical_seeds(self, question, scope, vector):
        entities, relations = {}, {}
        all_e, all_r = self.payloads('entity'), self.payloads('relation')
        for token in recall.lexical_tokens(question):
            variants = set(recall.identifier_variants(token))
            found = [e for e in all_e if live(e) and variants.intersection(
                [e.get('title'), *(e.get('aliases') or []), *(e.get('external_ids') or [])])]
            for e in found[:recall.LEXICAL_PER_TOKEN]:
                entities.setdefault(str(e['gr_id']), {**e, '_token': token,
                    '_score': recall.LEXICAL_SEED_SCORE['title' if e.get('title') in variants else 'alias']})
            predicate = lambda r: live(r) and in_scope(r, scope) and bool(variants.intersection([r.get('source'), r.get('target')]))
            found = (self.store.graph.vectors.search(self.collection('relation'), vector, recall.LEXICAL_PER_TOKEN * 2, predicate=predicate)
                     if vector is not None else [r for r in all_r if predicate(r)][:recall.LEXICAL_PER_TOKEN * 2])
            for r in found:
                relations.setdefault(str(r['gr_id']), {**r, '_token': token, '_score': recall.LEXICAL_SEED_SCORE['relation']})
        return list(entities.values()), list(relations.values())

    def lexical_facts(self, variants, limit, scope):
        return [r for r in self.payloads('spec') if live(r) and in_scope(r, scope)
                and (r.get('symbol') in variants or r.get('property') in variants)][:limit]

    def spec_dense(self, vector, limit, scope):
        merged = {}
        for name in recall.SPEC_VECTOR_NAMES:
            for r in self.dense('spec', vector, limit, scope, using=name):
                key = str(r.get('gr_id') or '')
                if not key:
                    continue
                if key not in merged:
                    merged[key] = {**r, '_named_hits': {name}, '_via': 'vector'}
                else:
                    merged[key]['_named_hits'].add(name)
                    merged[key]['_score'] = max(merged[key]['_score'], r['_score'])
        for r in merged.values():
            if {'property', 'value'} <= r.pop('_named_hits'):
                r['_score'] = min(1.0, r['_score'] + .05)
        return sorted(merged.values(), key=lambda r: -r['_score'])[:limit]

    def ranking_stats(self):
        return dict(max_rank=max((float(r.get('pagerank') or 0) for r in self.nodes('Entity')), default=0) or 1.0,
                    n_chunks=len(self.nodes('QdrantChunkSnapshot')),
                    total_mentions=self.con.execute("SELECT count(*) FROM edges WHERE kind='MENTIONED_IN'").fetchone()[0])

    def adjacent(self, ids, scope, *, direction='both', traversable=False):
        rows = self.con.execute("""SELECT e.id, r.payload, n.payload, x.src=e.uid
            FROM nodes e JOIN edges x ON x.kind='RELATED_TO' AND (x.src=e.uid OR x.dst=e.uid)
            JOIN nodes r ON r.uid=x.rel JOIN nodes n ON n.uid=CASE WHEN x.src=e.uid THEN x.dst ELSE x.src END
            WHERE e.kind='Entity' AND e.id IN (SELECT unnest(?)) ORDER BY e.id,r.id,n.id""", [list(ids)]).fetchall()
        out = []
        for eid, rel, other, outgoing in rows:
            r, n = json.loads(rel), json.loads(other)
            if r.get('boilerplate') or n.get('reference') or not in_scope(r, scope):
                continue
            if traversable and not r.get('traversable', False):
                continue
            if direction != 'both' and outgoing != (direction == 'out'):
                continue
            out.append((eid, r, n, outgoing))
        return sorted(out, key=lambda row: (-float(row[1].get('weight') or 0), -float(row[2].get('pagerank') or 0), str(row[1]['id']), str(row[2]['id'])))

    def expand(self, ids, limit, scope):
        out = defaultdict(list)
        for eid, r, n, _ in self.adjacent(ids, scope, traversable=True):
            if len(out[eid]) < limit:
                out[eid].append({**r, 'rid': r['id'], 'nid': n['id'], 'title': n.get('title'),
                    **{'n' + k: n.get(k) for k in ('type', 'upper', 'kind')},
                    **{k: n.get(k) for k in ('scope', 'pagerank', 'degree')}})
        return [{'eid': eid, 'neighbours': rows} for eid, rows in out.items()]

    def relation_cosines(self, vector, ids):
        from ..graph.vectors import point_id_for
        if vector is None:
            return {}
        mapping = {point_id_for(self.version, i): i for i in ids}
        return {mapping[pid]: recall._cosine(vector, vec) for pid, (_, vec) in self.store.graph.vectors.lookup(
            self.collection('relation'), list(mapping)).items()}

    def entity_titles(self, ids):
        return [dict(r, eid=r['id']) for r in self.nodes('Entity') if r['id'] in ids]

    def relation_titles(self, ids):
        return [dict(r, rid=r['id']) for r in self.nodes('Relation') if r['id'] in ids]

    def series_rows(self, keys, scope, doc_id=None):
        return sorted([r for r in self.nodes('Spec') if r.get('series_key') in keys and in_scope(r, scope)
                       and (not doc_id or r.get('doc_id') == doc_id)],
                      key=lambda r: (r['series_key'], r.get('series_index') or 0, r['id']))

    def document_paths(self, ids):
        return [r for r in self.nodes('Document') if r.get('doc_id') in ids]

    def mentions(self, ids):
        rows = self.con.execute("""SELECT e.id,c.payload,m.payload,
            (SELECT count(*) FROM edges x WHERE x.kind='MENTIONED_IN' AND x.dst=c.uid)
            FROM nodes e JOIN edges m ON m.src=e.uid AND m.kind='MENTIONED_IN'
            JOIN nodes c ON c.uid=m.dst WHERE e.id IN (SELECT unnest(?)) ORDER BY e.id,c.id""", [list(ids)]).fetchall()
        return [dict(json.loads(c), eid=eid, count=json.loads(m).get('count'), hub=hub)
                for eid, c, m, hub in rows if json.loads(c).get('kind') != 'boilerplate']

    def evidences(self, ids):
        rows = self.con.execute("""SELECT r.id,c.payload,tu.payload FROM nodes r
            JOIN edges ev ON ev.dst=r.uid AND ev.kind='EVIDENCES' JOIN nodes tu ON tu.uid=ev.src
            JOIN edges co ON co.dst=tu.uid AND co.kind='CONTRIBUTES_TO' JOIN nodes c ON c.uid=co.src
            WHERE r.id IN (SELECT unnest(?)) ORDER BY r.id,c.id""", [list(ids)]).fetchall()
        return [dict(json.loads(c), rid=rid, kind=json.loads(tu).get('kind') or 'body') for rid,c,tu in rows
                if json.loads(tu).get('kind') != 'boilerplate']

    def discovery_sources(self, entities, relations):
        for owners, rows, key in ((entities, self.mentions(list(entities)), 'eid'),
                                  (relations, self.evidences(list(relations)), 'rid')):
            for ident, slot in owners.items():
                found = sorted({(r.get('rel_path') or '', str(r['point_id'])) for r in rows if r[key] == ident})
                sample = found[:recall.SOURCE_SAMPLE]
                slot.update(point_ids=[p for _,p in sample], docs=sorted({d for d,_ in sample if d}),
                            sources_truncated=len(found) > recall.SOURCE_SAMPLE)

    def resolve(self, entity_id, name):
        from kb_search.graphwalk import _entity_row
        rows = self.nodes('Entity')
        if entity_id:
            exact = [r for r in rows if r['id'] == entity_id]
            rows = exact or [r for r in rows if entity_id in (r.get('ids') or [])][:5]
        else:
            normalize = lambda v: ''.join(str(v or '').casefold().split())
            name = normalize(name)
            rows = [r for r in rows if name and not r.get('boilerplate') and (
                any(normalize(v) == name for v in [r.get('title'), *(r.get('aliases') or [])]) or
                any(normalize(v) == name or normalize(v).endswith(':'+name) for v in (r.get('external_ids') or [])))]
            rows = sorted(rows, key=lambda r: (-float(r.get('pagerank') or 0),r['id']))[:10]
        return [_entity_row(r) for r in rows]

    def fact_subjects(self):
        rows = self.con.execute("""SELECT f.id,e.id FROM edges x JOIN nodes e ON x.src=e.uid
            JOIN nodes f ON x.dst=f.uid WHERE x.kind='HAS_SPEC' ORDER BY f.id,e.id""").fetchall()
        result = defaultdict(list)
        for fid, eid in rows:
            result[fid].append(eid)
        return result

    def fact_count(self, eid, scope):
        subjects = self.fact_subjects()
        return sum(eid in subjects[r['id']] and in_scope(r, scope) for r in self.nodes('Spec'))

    def neighbor_rows(self, eid, limit, types, direction, scope):
        from kb_search.graphwalk import RELATION_TIME_FIELDS, with_status
        out = []
        for _, r, n, outgoing in self.adjacent([eid], scope, direction=direction):
            if types and r.get('type') not in types:
                continue
            row = {'relation_id': r['id'], 'direction': 'out' if outgoing else 'in',
                   **{k: r.get(k) for k in ('type','directed','weight','npmi','cooccur','description','type_violation')},
                   **{k: r[k] for k in RELATION_TIME_FIELDS if r.get(k) not in (None, '')},
                   'other': {k:n.get(k) for k in ('id','title','type','upper','kind','scope','pagerank','degree')},
                   'evidence': [], '_names': [n.get('title'), *(n.get('aliases') or [])]}
            row['directed'] = bool(r.get('directed'))
            row['type_violation'] = bool(r.get('type_violation'))
            out.append(with_status(row, scope))
        return out[:limit]

    def neighbor_kinds(self, eid, direction, scope):
        counts = Counter(r.get('type') for _,r,_,_ in self.adjacent([eid], scope, direction=direction))
        return [{'type': k, 'count': v} for k,v in sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))]

    def attach_evidence(self, rows, limit):
        evidence = self.evidences([r['relation_id'] for r in rows])
        for r in rows:
            seen = {}
            for ev in evidence:
                if ev['rid'] == r['relation_id']:
                    seen.setdefault(str(ev['point_id']), {k: ev.get(k) for k in
                        ('point_id','chunk_uid','rel_path','doc_id','chunk_index','content_version','page_idx')})
            r['evidence'] = list(seen.values())[:limit]

    def entity_docs(self, eid, limit):
        return sorted({r['rel_path'] for r in self.mentions([eid]) if r.get('rel_path')})[:limit]

    domain_reader = True

    def dense_candidates(self, name):
        if not name:
            return []
        from ..embedding.client import EmbeddingClient
        from kb_search.graphwalk import CANDIDATE_LIMIT
        settings = self.store.settings
        vector = EmbeddingClient(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
            model_id=settings.embedding_model_id, dim=settings.embedding_dim, batch_size=1).embed([name])[0]
        return [dict(id=r['gr_id'], title=r.get('title'), type=r.get('type'), scope=r.get('scope'), score=r['_score'])
                for r in self.dense('entity', vector, CANDIDATE_LIMIT, None)]

    def list_entities(self, filters, offset, limit):
        from kb_search.graphwalk import LIST_DOCS_PER_ENTITY
        all_rows = [r for r in self.nodes('Entity') if live(r)]
        rows = []
        for r in all_rows:
            names = [r.get('title'), *(r.get('aliases') or []), *(r.get('external_ids') or [])]
            if filters['types'] and not set(filters['types']).intersection(r.get('type_path') or [str(r.get('type')).lower()]):
                continue
            if filters['uppers'] and str(r.get('upper') or 'entity').lower() not in filters['uppers']:
                continue
            if filters['kind'] != 'all' and (r.get('kind') or 'instance') != filters['kind']:
                continue
            if filters['name'] and not any(filters['name'] in str(n or '').casefold() for n in names):
                continue
            rows.append(r)
        rows.sort(key=lambda r: (-float(r.get('pagerank') or 0), r['id']))
        total, rows = len(rows), rows[offset:offset+limit]
        counts, classes = Counter(), Counter()
        for r in all_rows:
            key = (r.get('type'), r.get('upper') or 'entity')
            counts[key] += 1
            classes[key] += r.get('kind') == 'class'
        types = [dict(etype=k[0], upper=k[1], n=n, classes=classes[k])
                 for k,n in sorted(counts.items(), key=lambda x: (-x[1],str(x[0])))[:100]] if offset == 0 else None
        mentions = self.mentions([r['id'] for r in rows])
        docs = []
        for r in rows:
            paths = Counter(m['rel_path'] for m in mentions if m['eid'] == r['id'] and m.get('rel_path'))
            docs.append(dict(eid=r['id'], docs=[p for p,n in sorted(paths.items(), key=lambda x: (-x[1],x[0]))][:LIST_DOCS_PER_ENTITY], doc_count=len(paths)))
        return total, rows, types, docs

    def select_facts(self, center, needle, match, doc_id, conflicts, scope, offset, limit):
        from kb_search.graphwalk import CONCEPT_KEYS_MAX, PROPERTY_SUMMARY_LIMIT, SERIES_MEMBERS_MAX
        owners = self.fact_subjects()
        base = [r for r in self.nodes('Spec') if (center is None or center['id'] in owners[r['id']])
                and (not doc_id or r.get('doc_id') == doc_id)]
        rows = [r for r in base if not conflicts or r.get('conflict_group') is not None]
        matched, keys = None, []
        def matches(row, mode):
            words = [str(row.get(k) or '').lower() for k in ('property','symbol','concept')]
            return any(needle == w if mode == 'exact' else needle in w for w in words)
        if needle:
            for mode in (('exact','contains') if match == 'auto' else (match,)):
                found = [r for r in rows if matches(r, mode)]
                if found:
                    matched = mode
                    keys = list(dict.fromkeys(str(r['concept_key']) for r in found if r.get('concept_key')))[:CONCEPT_KEYS_MAX]
                    break
            rows = [r for r in rows if matched and (r.get('concept_key') in keys or matches(r, matched))]
        every = len(rows) if scope['mode'] != 'history' else None
        rows = [r for r in rows if in_scope(r, scope)]
        def order(r):
            return ((*((r.get('conflict_group') or '',) if conflicts else ()),
                     *((r.get('subject') or '',) if center is None else ()),
                     r.get('concept') or r.get('property') or '', r.get('valid_from') or '', r.get('series_index') or 0, r['id']))
        rows.sort(key=order)
        total, picked = len(rows), rows[offset:offset+limit]
        subjects = [center['id'] if center else next(iter(owners[r['id']]), None) for r in picked]
        series = defaultdict(list)
        for r in self.series_rows({r.get('series_key') for r in picked if r.get('series_key')}, scope, doc_id):
            if len(series[r['series_key']]) < SERIES_MEMBERS_MAX:
                series[r['series_key']].append({**{k:r.get(k) for k in ('id','value','min','typ','max','unit','unit_canonical','valid_from','valid_until','time_basis','observed_at','observed','series_index')},
                    'when': r.get('valid_from') or r.get('observed_at') or r.get('axis') or r.get('observed')})
        properties = None
        if (center or doc_id) and offset == 0:
            counts = Counter((r.get('concept') or r.get('property'), r.get('concept_key')) for r in base if in_scope(r, scope))
            properties = [dict(concept=k[0],concept_key=k[1],n=n) for k,n in sorted(counts.items(),key=lambda x:(-x[1],str(x[0])))[:PROPERTY_SUMMARY_LIMIT]]
        return picked, subjects, dict(series), properties, matched, keys, every, total
