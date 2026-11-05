"""Scores and labels used by the paper's evaluation protocol."""


def qa_score(result):
    score = result.get('score') if isinstance(result, dict) else None
    if type(score) not in (int, float) or score not in (0, 0.5, 1):
        raise ValueError('QA score must be 0, 0.5, or 1')
    return float(score)


def compute_nugget_metrics(coverage, nuggets):
    ids = [n['nugget_id'] for n in nuggets]
    if not ids or not isinstance(coverage, dict) or set(coverage) != set(ids):
        raise ValueError('SUM coverage must contain exactly the reference nugget IDs')
    if any(type(v) not in (bool, int, float) or v not in (0, 1) for v in coverage.values()):
        raise ValueError('SUM labels must be 0 or 1')
    coverage = {key: int(coverage[key]) for key in ids}
    covered = sum(coverage.values())
    result = {
        'nugget_coverage': coverage,
        'nugget_recall': covered / len(ids),
        'nugget_total': len(ids),
        'nugget_covered': covered,
    }
    mm_ids = [n['nugget_id'] for n in nuggets if n.get('source_type') == 'multimodal']
    if mm_ids:
        result['mm_nugget_recall'] = sum(coverage[key] for key in mm_ids) / len(mm_ids)
    return result


def retrieval_judgment(result, clue_count, entry_count):
    if not isinstance(result, dict):
        raise ValueError('Retrieval judgment must be a JSON object')
    parsed = {'reasoning': result.get('reasoning', '')}
    for key, count in [('clue_coverage', clue_count), ('entry_relevance', entry_count)]:
        labels = result.get(key)
        if not isinstance(labels, list) or len(labels) != count:
            raise ValueError(f'{key} must contain {count} labels')
        if any(type(v) not in (bool, int, float) or v not in (0, 1) for v in labels):
            raise ValueError(f'{key} labels must be boolean or 0/1')
        parsed[key] = [bool(v) for v in labels]
    return parsed
