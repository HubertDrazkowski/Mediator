"""Small checks for empirical replay; no dataset download or learner sweep."""
import csv

import numpy as np
import pytest

from replication.star import (ACTION_LABELS, normalize_rows, prepare_star_rows,
                              read_star_rows, rows_sha256, sample_empirical_row,
                              star_rng_seeds)


def toy_rows():
    # The reference has tied later scores. Extra rows test CDF extrapolation
    # and retention of missing early achievement without inventing outcomes.
    rows = []
    for a, label in enumerate(ACTION_LABELS):
        for z in range(8):
            rows.append(dict(row_id=f'{a}-{z}', action=label,
                             mediator_value=float(z), outcome_value=float(z // 2 + a)))
        rows.append(dict(row_id=f'{a}-missing', action=label,
                         mediator_value=None, outcome_value=float(3 + a)))
        rows.append(dict(row_id=f'{a}-no-outcome', action=label,
                         mediator_value=20., outcome_value=None))
    rows.append(dict(row_id='no-action', action=None, mediator_value=1., outcome_value=2.))
    return rows


def test_star_missing_state_frozen_reference_and_joint_law():
    rows = toy_rows()
    instance = prepare_star_rows(rows, horizon=100)
    meta = instance['metadata']
    assert (meta['n_source'], meta['n_retained'], meta['n_loss_reference'], meta['n_missing_mediator']) == (31, 27, 24, 3)
    observed = [row['mediator_value'] for row in rows if row['action'] is not None and row['mediator_value'] is not None]
    np.testing.assert_array_equal(instance['mediator_edges'], np.quantile(observed, [.25, .5, .75]))
    reference = np.sort([row['outcome_value'] for row in rows
                         if row['action'] is not None and row['outcome_value'] is not None
                         and row['mediator_value'] is not None])
    retained = [row for row in rows if row['action'] is not None and row['outcome_value'] is not None]
    expected_loss = np.array([1 - (np.sum(reference < row['outcome_value']) + .5 * np.sum(reference == row['outcome_value'])) / len(reference)
                              for row in retained])
    np.testing.assert_array_equal(instance['empirical_loss'], expected_loss)
    np.testing.assert_array_equal(instance['empirical_row_id'], [row['row_id'] for row in retained])
    assert np.all(instance['empirical_mediator'][[8, 17, 26]] == 4)
    for a in range(3):
        selected = instance['empirical_action'] == a
        assert instance['empirical_mu'][a] == expected_loss[selected].mean()
        np.testing.assert_array_equal(instance['M'][:, a], np.bincount(instance['empirical_mediator'][selected], minlength=5) / selected.sum())
    assert np.max(np.abs(instance['M'].T @ instance['vectors'][0] - instance['empirical_mu'])) > .01
    assert meta['full_mediation_enforced'] is False


def test_star_midpoint_cdf_does_not_rescale_added_rows():
    rows = toy_rows()
    baseline = prepare_star_rows(rows)
    modified = [dict(row) for row in rows]
    modified[-2]['outcome_value'] = None
    # Add extreme later scores with missing early measurement. They cannot
    # alter the CDF reference or any previously retained student's loss.
    modified.extend([dict(row_id='extra-low', action='small', mediator_value=None, outcome_value=-1.),
                     dict(row_id='extra-high', action='small', mediator_value=None, outcome_value=100.)])
    extended = prepare_star_rows(modified)
    lookup = dict(zip(extended['empirical_row_id'], extended['empirical_loss']))
    for identifier, loss in zip(baseline['empirical_row_id'], baseline['empirical_loss']):
        assert lookup[identifier] == loss
    assert lookup['extra-low'] == 1.
    assert lookup['extra-high'] == 0.


def test_star_ties_enter_upper_bin_and_sampler_keeps_row_order():
    instance = prepare_star_rows(toy_rows())
    for edge in instance['mediator_edges']:
        exact = instance['empirical_mediator_value'] == edge
        if exact.any():
            assert np.all(instance['empirical_mediator'][exact] == np.searchsorted(instance['mediator_edges'], edge, side='right'))
    indices = [np.flatnonzero(instance['empirical_action'] == a) for a in range(3)]
    for a in range(3):
        assert sample_empirical_row(indices, a, 0.) == indices[a][0]
        assert sample_empirical_row(indices, a, np.nextafter(1., 0.)) == indices[a][-1]
        assert sample_empirical_row(indices, a, .5) == indices[a][len(indices[a]) // 2]
    with pytest.raises(ValueError):
        sample_empirical_row(indices, 0, 1.)


def test_star_source_hash_checks_order_and_public_field_types(tmp_path):
    rows = normalize_rows(toy_rows())
    path = tmp_path / 'STAR.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    decoded = read_star_rows(path)
    assert rows_sha256(decoded) == rows_sha256(rows)
    assert rows_sha256(decoded[::-1]) != rows_sha256(rows)
    with pytest.raises(ValueError, match='differ from the frozen'):
        prepare_star_rows(rows, verify_paper=True)
    with pytest.raises(ValueError, match='distinct'):
        normalize_rows(rows + [rows[0]])


def test_star_rng_matches_saved_encoding():
    # Golden stream IDs ensure tuple spacing/encoding is not accidentally changed.
    environment, learner = star_rng_seeds(27001)
    assert environment == 181885811566006733734541353505052539982
    assert learner == 208622422863128743573325866285938995809
