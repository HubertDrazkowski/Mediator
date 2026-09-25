"""Stationary STAR whole-record replay, preserving the observed joint law.

Loading is local and never downloads data. Run ``scripts/fetch_star.py`` once
to obtain the public source and produce the small, documented CSV interface.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_URL = 'https://raw.githubusercontent.com/cran/AER/master/data/STAR.rda'
DOCUMENTATION_URL = 'https://cran.r-universe.dev/AER/doc/manual.html#STAR'
SOURCE_SHA256 = '9e597f8096550185acb7917e6b087a45815574b0f2a4d770222395dd85a4b7b1'
ACTION_LABELS = ['regular', 'regular+aide', 'small']
MEDIATOR_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Missing K score']
REQUIRED_COLUMNS = ('row_id', 'action', 'mediator_value', 'outcome_value')
# Ordered semantic contents of the public CSV; independent of CSV line endings.
PAPER_ROWS_SHA256 = '942180c2c1891b9c566ddd7c3d8633bc16e55397048d0c9fa71b975f9f301dcb'


def _finite_number(value):
    if value is None or str(value).strip() in ('', 'NA', 'NaN', 'nan'):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def normalize_rows(rows):
    """Keep only the four public fields, with stable types and source ordering."""
    normalized = []
    identifiers = set()
    for row in rows:
        if any(key not in row for key in REQUIRED_COLUMNS):
            raise ValueError('STAR rows need: ' + ', '.join(REQUIRED_COLUMNS))
        identifier = str(row['row_id'])
        if not identifier or identifier in identifiers:
            raise ValueError('STAR requires distinct, nonempty source row IDs')
        identifiers.add(identifier)
        action = row['action']
        action = None if action is None or str(action).strip() in ('', 'NA') else str(action)
        if action is not None and action not in ACTION_LABELS:
            raise ValueError('Unrecognized STAR kindergarten action: ' + action)
        normalized.append(dict(row_id=identifier, action=action,
                               mediator_value=_finite_number(row['mediator_value']),
                               outcome_value=_finite_number(row['outcome_value'])))
    return normalized


def rows_sha256(rows):
    """Hash ordered source IDs and values, not incidental CSV formatting."""
    payload = [[row[key] for key in REQUIRED_COLUMNS] for row in rows]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_star_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return normalize_rows(csv.DictReader(stream))


def prepare_star_rows(rows, horizon=20000, verify_paper=False, source_metadata=None):
    """Build one empirical environment from public rows without writing data.

    Quantile cutpoints use every known-arm/observed-kindergarten-score record.
    The loss reference uses the original cohort with both scores observed.
    Missing kindergarten scores are retained as state 4; missing later scores
    remain excluded. Neither arm means nor losses are pooled across actions
    when sampling or measuring pseudo-regret.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    rows = normalize_rows(rows)
    semantic_hash = rows_sha256(rows)
    if verify_paper and semantic_hash != PAPER_ROWS_SHA256:
        raise ValueError('STAR source rows differ from the frozen paper source; run scripts/fetch_star.py')
    known = [row for row in rows if row['action'] is not None]
    observed = [row for row in known if row['mediator_value'] is not None]
    reference = [row for row in observed if row['outcome_value'] is not None]
    retained = [row for row in known if row['outcome_value'] is not None]
    if not observed or not reference or not retained:
        raise ValueError('STAR requires observed early and later scores in the reference cohort')
    edges = np.unique(np.quantile([row['mediator_value'] for row in observed], [.25, .5, .75]))
    if len(edges) != 3:
        raise ValueError('STAR quartile construction requires three distinct cutpoints')
    actions = np.array([ACTION_LABELS.index(row['action']) for row in retained], dtype=np.int16)
    med = np.array([np.nan if row['mediator_value'] is None else row['mediator_value'] for row in retained])
    outcome = np.array([row['outcome_value'] for row in retained], dtype=np.float64)
    mediators = np.where(np.isfinite(med), np.searchsorted(edges, med, side='right'), 4).astype(np.int16)
    reference_scores = np.sort(np.array([row['outcome_value'] for row in reference], dtype=np.float64))
    left = np.searchsorted(reference_scores, outcome, side='left')
    right = np.searchsorted(reference_scores, outcome, side='right')
    losses = 1 - (left + .5 * (right - left)) / len(reference_scores)
    counts = np.zeros((5, 3), dtype=np.int64)
    cell_means = np.full((5, 3), np.nan)
    for a in range(3):
        for z in range(5):
            selected = (actions == a) & (mediators == z)
            counts[z, a] = selected.sum()
            if selected.any():
                cell_means[z, a] = losses[selected].mean()
    if np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0):
        raise ValueError('The STAR benchmark requires every action and mediator state to be represented')
    M = counts / counts.sum(axis=0, keepdims=True)
    mu = np.array([losses[actions == a].mean() for a in range(3)])
    pooled_g = np.array([losses[mediators == z].mean() for z in range(5)])
    if verify_paper:
        assert len(rows) == 11598 and len(known) == 6325 and len(observed) == 5786
        assert len(reference) == 3999 and len(retained) == 4298
        assert np.sum(mediators == 4) == 299
        np.testing.assert_array_equal(edges, [435., 457.5, 482.])
        assert np.all(counts > 0)
    gaps = mu - mu.min()
    metadata = dict(
        family='STAR', experiment='STAR', dataset='STAR', mode='stationary',
        scenario_id='STAR_missingmediator_v1', sampling='empirical_rows',
        n_actions=3, n_contexts=5, epsilon=None, horizon=horizon,
        action_labels=ACTION_LABELS, mediator_labels=MEDIATOR_LABELS,
        source_url=SOURCE_URL, source_documentation=DOCUMENTATION_URL,
        expected_source_rda_sha256=SOURCE_SHA256, source_rows_sha256=semantic_hash,
        n_source=len(rows), n_known_action=len(known), n_observed_mediator=len(observed),
        n_loss_reference=len(reference), n_retained=len(retained),
        n_missing_mediator=int(np.sum(mediators == 4)),
        n_excluded_outcome=len(known) - len(retained),
        mediator_edges=edges.tolist(), cell_counts=counts.tolist(),
        action_counts=counts.sum(axis=0).tolist(), action_means=mu.tolist(),
        action_gaps=gaps.tolist(), optimal_actions=np.flatnonzero(gaps <= 1e-12).tolist(),
        uniform_regret={str(t): float(t * gaps.mean()) for t in sorted({min(5000, horizon), horizon})},
        mechanism_fixed_over_time=True, mechanism_is_oracle_for_empirical_environment=True,
        full_mediation_enforced=False, no_algorithm_performance_selection=True,
        cohort_selection='Observed kindergarten class type and both first-grade scores; missing early scores retained',
        loss_definition='1 - fixed midpoint empirical CDF of first-grade score; reference is the original complete-case cohort',
        regret_reference='Raw empirical per-action means, not M.T @ pooled mediator means',
        sampling_description='Whole retained row uniformly with replacement within selected action; original source order',
        action_complexity=float(np.sum(1 / gaps[gaps > 1e-12])),
        context_complexity=None, S_Z_ach=None, complexity_relation='not_applicable_under_nonseparation',
        phase_diagnostics=[],
    )
    if source_metadata:
        metadata.update(source_metadata)
    return dict(
        scenario_id=metadata['scenario_id'], M=M, vectors=np.array([pooled_g]),
        starts=np.array([1], dtype=np.int32), baseline_groups=np.arange(5),
        empirical_action=actions, empirical_mediator=mediators, empirical_loss=losses,
        empirical_mu=mu, empirical_row_id=np.array([row['row_id'] for row in retained]),
        empirical_mediator_value=med, empirical_outcome_value=outcome,
        cell_means=cell_means, mediator_edges=edges, metadata=metadata,
    )


def load_star_instance(source_path='data/STAR.csv', horizon=20000, verify_paper=True):
    """Load an explicitly fetched local CSV; never perform an implicit download."""
    from .io import safe_relative
    path = safe_relative(source_path)
    if not path.is_file():
        raise FileNotFoundError('STAR data are not bundled. Run: python scripts/fetch_star.py')
    provenance = dict(source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return prepare_star_rows(read_star_rows(path), horizon, verify_paper, provenance)


def star_rng_seeds(seed):
    """Return the exact historical environment and learner random-stream IDs."""
    from .io import stable_seed
    return stable_seed('raw_realdata_v1', 'STAR', seed), stable_seed('STAR', seed, 'learner')


def sample_empirical_row(action_indices, action, uniform):
    """Inverse-CDF uniform row selection; use the original retained row order."""
    if not 0 <= uniform < 1:
        raise ValueError('The row uniform must lie in [0,1)')
    indices = action_indices[action]
    return int(indices[int(uniform * len(indices))])
