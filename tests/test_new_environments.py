"""Generator semantics and frozen-paper input fingerprints, without saved datasets.

The four fingerprints were independently checked against all historical NPZ
inputs: 90 E1 schedules, six transpose settings, 70 stationary random graphs
and their 30 hash-selected switching counterparts. Tests regenerate only
environments, never learner trajectories.
"""
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from replication.environments import (
    generate_e1_piecewise, generate_random_paper, generate_random_switching,
    generate_transpose,
)
from replication import configuration, runner


def input_fingerprint(instances):
    """Stable ID plus the four historical input arrays, in ID order."""
    named = [(v['metadata'].get('rng_scenario_id', v['scenario_id']), v) for v in instances]
    digest = hashlib.sha256()
    for sid, instance in sorted(named):
        digest.update(sid.encode())
        for key in ('M', 'vectors', 'starts', 'baseline_groups'):
            digest.update(np.ascontiguousarray(instance[key]).tobytes())
    return digest.hexdigest()


class PiecewiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = [generate_e1_piecewise(a, z, seed)
                         for a, z in ((80, 30), (30, 80), (60, 60))
                         for seed in range(8201, 8231)]

    def test_all_thirty_seed_specific_schedules_match_frozen_inputs(self):
        self.assertEqual(len(self.instances), 90)
        self.assertEqual(input_fingerprint(self.instances),
                         '44fe6f51e32ef7d12ce2de68477370e1c5ea6300f6c981f02861d5685c5e2cf5')

    def test_common_shift_preserves_gaps_and_is_shared_across_dimensions(self):
        by_seed = {}
        for instance in self.instances:
            M, vectors, meta = instance['M'], instance['vectors'], instance['metadata']
            self.assertFalse(meta['clipping_active'])
            self.assertFalse(meta['learner_reset_at_switch'])
            self.assertEqual(instance['trajectory_seed'], meta['replicate_seed'])
            seed = instance['trajectory_seed']
            by_seed.setdefault(seed, meta['beta_values'])
            self.assertEqual(by_seed[seed], meta['beta_values'])
            self.assertEqual(len(vectors), 13)
            expected_gaps = .07 * (np.arange(M.shape[1]) % M.shape[0] != 0)
            for vector in vectors:
                mu = M.T @ vector
                np.testing.assert_allclose(mu - mu.min(), expected_gaps, rtol=0, atol=2e-14)
            self.assertTrue(np.all((vectors >= 0) & (vectors <= 1)))
        self.assertEqual(len({tuple(v) for v in by_seed.values()}), 30)

    def test_zero_is_a_valid_trajectory_seed(self):
        instance = generate_e1_piecewise(3, 4, 0, horizon=100)
        self.assertEqual(instance['trajectory_seed'], 0)


class TransposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = [generate_transpose(n, orientation)
                         for n in (8, 12, 16) for orientation in ('vertex', 'edge')]

    def test_six_complete_graph_inputs_match_frozen_inputs(self):
        self.assertEqual(input_fingerprint(self.instances),
                         '7014df1545b2f8ccdc3d0a69f00ad48393f5cd8468079bc2ea054884e2d75952')

    def test_analytic_joint_margin_primal_and_dual_certificates(self):
        for instance in self.instances:
            M, g, meta = instance['M'], instance['vectors'][0], instance['metadata']
            mu = M.T @ g
            gaps = mu - mu.min()
            self.assertEqual(len(np.flatnonzero(gaps <= 1e-12)), 1)
            self.assertAlmostEqual(float(gaps[gaps > 1e-12].min()), .02)
            self.assertAlmostEqual(float(np.sum(1 / gaps[gaps > 1e-12])), meta['S_A'])
            B = M[np.asarray(meta['opening_contexts'])]
            margins = np.asarray(meta['context_substitution_margins'])
            dual = np.asarray(meta['dual_action_weights'])
            self.assertTrue(np.all(B.T @ margins <= gaps + 1e-14))
            self.assertAlmostEqual(float(np.sum(1 / margins)), meta['S_Z_ach'])
            dual_value = 2 * np.sqrt(B @ dual).sum() - gaps @ dual
            self.assertAlmostEqual(float(dual_value), meta['S_Z_ach'], places=7)
            ratio = meta['S_A'] / meta['S_Z_ach']
            if meta['orientation'] == 'vertex':
                self.assertLess(ratio, 1)
            else:
                self.assertGreater(ratio, 1)


class RandomExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stationary = generate_random_paper(joint_diagnostics=False)
        cls.switching = generate_random_switching(cls.stationary)

    def test_independent_fifty_plus_twenty_batches_match_frozen_inputs(self):
        self.assertEqual(len(self.stationary), 70)
        self.assertEqual(sum(v['metadata']['sample_cohort'] == 'original' for v in self.stationary), 50)
        self.assertEqual(sum(v['metadata']['sample_cohort'] == 'extension' for v in self.stationary), 20)
        self.assertEqual(input_fingerprint(self.stationary),
                         '7dd6c1bd9079b32c70ba69bcefee5f19e84bd7348266c9461828e53e5413d3e0')

    def test_hash_selected_switching_inputs_and_paired_rng_identifiers(self):
        self.assertEqual(len(self.switching), 30)
        self.assertEqual(input_fingerprint(self.switching),
                         'a92fdcd4cb96948067c86dbfa230275c4530cdc83de48319af924b85de52ba33')
        by_id = {v['scenario_id']: v for v in self.stationary}
        rank = lambda sid: hashlib.sha256(f'2026092401|{sid}'.encode()).hexdigest()
        selected = sorted(by_id, key=rank)[:30]
        self.assertEqual([v['metadata']['rng_scenario_id'] for v in self.switching], selected)
        self.assertEqual(sum(v['metadata']['matched_switching'] for v in self.stationary), 30)
        self.assertFalse(set(by_id) & {v['scenario_id'] for v in self.switching})
        for instance in self.switching:
            metadata = instance['metadata']
            source = by_id[metadata['rng_scenario_id']]
            self.assertEqual(metadata['switching_source_scenario_id'], source['scenario_id'])
            M, g0 = source['M'], source['vectors'][0]
            np.testing.assert_array_equal(instance['M'], M)
            mu0 = M.T @ g0
            for vector in instance['vectors']:
                self.assertTrue(np.all((vector >= 0) & (vector <= 1)))
                mu = M.T @ vector
                np.testing.assert_allclose(mu - mu.min(), mu0 - mu0.min(), rtol=0, atol=2e-15)
                self.assertLess(float(np.ptp(vector - g0)), 3e-16)

    def test_no_isolated_vertices_or_cross_community_edges_when_r_is_one(self):
        for instance in self.stationary:
            M, meta = instance['M'], instance['metadata']
            self.assertTrue(np.all(np.any(M > 0, axis=0)))
            self.assertTrue(np.all(np.any(M > 0, axis=1)))
            np.testing.assert_allclose(M.sum(axis=0), 1., rtol=0, atol=1e-14)
            if meta['r'] == 1:
                different = (np.asarray(meta['context_blocks'])[:, None]
                             != np.asarray(meta['action_blocks'])[None, :])
                self.assertTrue(np.all(M[different] == 0))

    def test_zero_extension_and_zero_switching_allow_stationary_only(self):
        stationary = generate_random_paper(horizon=100, joint_diagnostics=False,
            config=dict(n_graphs=2, dimensions=[(3, 4)], extension_n_graphs=0))
        self.assertEqual(len(stationary), 2)
        self.assertEqual(generate_random_switching(stationary, count=0, horizon=100), [])
        self.assertFalse(any(v['metadata']['matched_switching'] for v in stationary))


class PlanIntegrationTests(unittest.TestCase):
    def prepare_temporary(self, config, temporary):
        with patch.object(runner, 'safe_relative', lambda value: Path(temporary) / value), redirect_stdout(io.StringIO()):
            output = runner.prepare(config)
        return output, json.loads((output / 'manifest.json').read_text())

    def test_piecewise_schedule_runs_only_its_matching_trajectory_seed(self):
        config = configuration.load_config('configs/e1.json')
        config.update(dimensions=[[3, 4]], seeds=[8201, 8202], horizon=9,
                      checkpoints=[9], modes=['random_baseline_piecewise'],
                      algorithms=['CTsallis-Action'])
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest = self.prepare_temporary(config, temporary)
            self.assertEqual(len(manifest['instances']), 2)
            self.assertEqual(len(manifest['tasks']), 2)
            for task in manifest['tasks']:
                metadata = json.loads((output / 'instances' / (task['scenario_id'] + '.json')).read_text())
                self.assertEqual(task['seed'], metadata['trajectory_seed'])

    def test_switching_has_distinct_storage_identity_and_paired_rng_identity(self):
        config = configuration.load_config('configs/random.json')
        config.update(horizon=9, checkpoints=[9], algorithms=['CTsallis-Action'])
        config['random'].update(n_graphs=2, extension_n_graphs=0, switching_count=1,
                                dimensions=[[3, 4]], joint_diagnostics=False)
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest = self.prepare_temporary(config, temporary)
            self.assertEqual(len(manifest['instances']), 3)
            self.assertEqual(len(manifest['tasks']), 3)
            switching = next(t for t in manifest['tasks'] if t['scenario_id'].endswith('__switching'))
            paired = next(t for t in manifest['tasks'] if t['scenario_id'] == switching['rng_scenario_id'])
            self.assertNotEqual(switching['task_id'], paired['task_id'])
            self.assertEqual(switching['seed'], paired['seed'])
            self.assertEqual(switching['rng_scenario_id'], paired['rng_scenario_id'])
            self.assertTrue((output / 'instances' / (switching['scenario_id'] + '.npz')).exists())


if __name__ == '__main__':
    unittest.main()
