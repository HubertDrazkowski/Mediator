"""Code-only semantic checks; all observations are generated in temporary folders.

Run from the package root: python -m unittest discover -s tests -v
These short tests check the pipeline, not the statistical conclusions of the paper.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from replication import algorithms, calibration, configuration, environments, runner
from replication import io as result_io


class EnvironmentTests(unittest.TestCase):
    def test_e1_mechanism_and_constant_gaps(self):
        for A, Z in ((8, 3), (3, 8), (6, 6)):
            stationary, switching = environments.generate_e1(A, Z, horizon=300)
            M = stationary['M']
            expected = np.full((Z, A), .3 / Z)
            expected[np.arange(A) % Z, np.arange(A)] += .7
            np.testing.assert_array_equal(M, expected)
            np.testing.assert_allclose(M.sum(axis=0), 1, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(M, switching['M'])
            expected_gaps = .07 * (np.arange(A) % Z != 0)
            for instance in (stationary, switching):
                self.assertTrue(np.all((instance['vectors'] >= 0) & (instance['vectors'] <= 1)))
                for g in instance['vectors']:
                    means = M.T @ g
                    np.testing.assert_allclose(means - means.min(), expected_gaps, atol=1e-14)
                # Positive epsilon makes every mediator reachable from an optimum.
                self.assertEqual(instance['metadata']['context_complexity'], 0)
            self.assertEqual(stationary['metadata']['unique_columns'], min(A, Z))

    def test_switch_schedule_uses_iterative_rounding(self):
        # The third length is 128; exponentiation can round it upward to 129.
        np.testing.assert_array_equal(environments.geometric_starts(300), [1, 51, 131, 259])
        starts = environments.geometric_starts(1000)
        self.assertEqual(starts[0], 1)
        self.assertTrue(np.all(np.diff(starts) > 0))
        self.assertLessEqual(starts[-1], 1000)

    def test_joint_attainable_complexity_shares_one_action_budget(self):
        M = np.array([[1., 0.], [0., .5], [0., .5]])
        result = environments.complexity_diagnostics(M, np.array([.1, .3, .3]))
        self.assertAlmostEqual(result['action_complexity'], 5.)
        # Both mediator margins must share the second action's 0.2 gap.
        self.assertAlmostEqual(result['context_complexity'], 10., places=5)
        self.assertEqual(result['complexity_relation'], 'SA_lt_SZ')

    def test_random_generation_is_reproducible_and_column_stochastic(self):
        config = dict(seed=17, horizon=7, n_graphs=2, losses_per_graph=1,
                      dimensions=[(3, 4)], joint_diagnostics=False)
        first, second = environments.generate_random(config), environments.generate_random(config)
        self.assertEqual(len(first), 2)
        for a, b in zip(first, second):
            self.assertEqual(a['scenario_id'], b['scenario_id'])
            for key in ('M', 'vectors', 'starts', 'baseline_groups'):
                np.testing.assert_array_equal(a[key], b[key])
            self.assertTrue(np.all(a['M'] >= 0))
            np.testing.assert_allclose(a['M'].sum(axis=0), 1., atol=1e-14)
            self.assertTrue(np.all((a['vectors'] >= 0) & (a['vectors'] <= 1)))


class SelectionTests(unittest.TestCase):
    def test_paper_selections_and_method_counts(self):
        for experiment, methods in (('e1', 10), ('random', 10), ('online', 11)):
            cfg = configuration.load_config('configs/' + experiment + '.json')
            specs = configuration.selected_specs(cfg)
            self.assertEqual(len(specs), methods)
            self.assertEqual(cfg['horizon'], 30000)
            self.assertEqual(cfg['checkpoints'], [5000, 30000])
            self.assertEqual('EXP4MF-RV' in {s['id'] for s in specs}, experiment == 'online')
            if experiment != 'random':
                self.assertEqual(cfg['seeds'], list(range(8201, 8231)))
                self.assertEqual(cfg['dimensions'], [[80, 30], [30, 80], [60, 60]])
                self.assertEqual(cfg['epsilons'], [.3])
                self.assertEqual(set(cfg['modes']), {'stationary', 'baseline_switching', 'random_baseline_piecewise'} if experiment=='e1' else {'stationary','baseline_switching'})
            else:
                self.assertEqual(cfg['seeds'], [8201])
                self.assertEqual(cfg['random']['n_graphs'], 50)
                self.assertEqual(cfg['random']['extension_n_graphs'], 20)
                self.assertEqual(cfg['random']['switching_count'], 30)
            if experiment == 'online':
                self.assertEqual(cfg['initial_pairs'], [500, 2000])

    def test_aliases_and_overrides_do_not_mutate_defaults(self):
        cfg = configuration.load_config('configs/e1.json')
        cfg['algorithms'] = ['CTsallis-Mediator']
        cfg['algorithm_overrides'] = {'CTsallis-Context': {'c': .9}}
        self.assertEqual(configuration.selected_specs(cfg)[0]['c'], .9)
        default = next(s for s in algorithms.default_specs() if s['id'] == 'CTsallis-Context')
        self.assertEqual(default['c'], .7)
        cfg['algorithms'] += ['CTsallis-Context']
        with self.assertRaises(ValueError):
            configuration.selected_specs(cfg)
        cfg['algorithms'] = ['CTsallis-INF (no Hα)']
        cfg['algorithm_overrides'] = {'CTsallis-Jensen-rho0': {'rho': .01}}
        with self.assertRaises(ValueError):
            configuration.selected_specs(cfg)

    def test_smoke_is_distinct_and_does_not_mutate_paper_config(self):
        paper = configuration.load_config('configs/online.json')
        original = deepcopy(paper)
        smoke = configuration.smoke_config(paper)
        self.assertEqual(paper, original)
        self.assertEqual(smoke['horizon'], 200)
        self.assertNotEqual(smoke['output'], paper['output'])
        self.assertEqual(len(smoke['seeds']), 1)

    def test_output_paths_must_be_internal_relative_paths(self):
        for invalid in ('', '.', '../outside', '/absolute/output', 'results/../../outside'):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                result_io.safe_relative(invalid)
        self.assertEqual(result_io.safe_relative('results/local').parent, result_io.ROOT / 'results')


class OnlineTests(unittest.TestCase):
    def test_calibration_is_nested_and_respects_conditional_mechanism(self):
        M = np.eye(3)
        a, z = calibration.sample_calibration(M, 19, n_max=20)
        small_a, small_z = calibration.sample_calibration(M, 19, n_max=7)
        np.testing.assert_array_equal(a, z)
        np.testing.assert_array_equal(a[:7], small_a)
        np.testing.assert_array_equal(z[:7], small_z)
        counts = calibration.count_pairs(a, z, 3, 3, n=7)
        self.assertEqual(counts.sum(), 7)
        self.assertEqual(np.count_nonzero(counts - np.diag(np.diag(counts))), 0)

    def test_estimator_uses_only_completed_observations_and_immutable_snapshots(self):
        counts = np.array([[2, 0], [0, 3]], dtype=np.int64)
        estimator = calibration.OnlineMechanismEstimate(counts)
        with self.assertRaises(ValueError):
            estimator.observe(1, 0, 1)
        before = estimator.matrix(1)
        saved = before.copy()
        with self.assertRaises(ValueError):
            before[0, 0] = 0
        estimator.observe(1, 0, 1)
        after = estimator.matrix(2)
        np.testing.assert_array_equal(before, saved)
        np.testing.assert_array_equal(after[:, 1], before[:, 1])
        expected_counts = counts.copy()
        expected_counts[1, 0] += 1
        np.testing.assert_array_equal(estimator.counts, expected_counts)
        np.testing.assert_allclose(after, (expected_counts + .5) / (expected_counts.sum(axis=0) + 1))
        with self.assertRaises(ValueError):
            estimator.observe(1, 0, 0)
        with self.assertRaises(ValueError):
            estimator.matrix(3)

    def test_online_model_refresh_preserves_past_action_estimates(self):
        M = np.array([[.8, .2], [.2, .8]])
        changed = np.array([[.3, .6], [.7, .4]])
        for name in ('CTsallis-Action', 'CTsallis-Context', 'CTsallis-Jensen-selected', 'CTsallis-Jensen-rho0'):
            with self.subTest(algorithm=name):
                learner = algorithms.instantiate(name, M, 20, np.random.default_rng(3), online=True)
                p = learner.probabilities(1)
                learner.update(1, 0, 0, 1., p)
                history = learner.cumulative_action_estimates.copy()
                self.assertGreater(np.ptp(history), 0)
                learner.update_mechanism(changed)
                np.testing.assert_array_equal(learner.cumulative_action_estimates, history)
                with self.assertRaises(ValueError):
                    learner.update_mechanism(M)
                next_p = learner.probabilities(2)
                self.assertAlmostEqual(float(next_p.sum()), 1.)

    def test_all_methods_return_simplex_and_preserve_input_matrix(self):
        M = environments.generate_e1(3, 4, horizon=5)[0]['M']
        original = M.copy()
        for spec in algorithms.default_specs():
            for online in (False, True):
                with self.subTest(algorithm=spec['id'], online=online):
                    learner = algorithms.instantiate(spec, M, 5, np.random.default_rng(1), online=online)
                    p = learner.probabilities(1)
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue(np.all(p >= 0))
                    self.assertAlmostEqual(float(p.sum()), 1.)
                    # PE commits to a scheduled action; honor its point mass.
                    learner.update(1, int(np.argmax(p)), 0, 1., p)
                    if online:
                        learner.update_mechanism(M)
                    p = learner.probabilities(2)
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue(np.all(p >= 0))
                    self.assertAlmostEqual(float(p.sum()), 1.)
        np.testing.assert_array_equal(M, original)
        self.assertIsNotNone(algorithms.applicability('CUCB2', np.eye(3)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, experiment='e1'):
        cfg = configuration.load_config('configs/' + experiment + '.json')
        cfg.update(horizon=9, checkpoints=[4, 9], jobs=1, output='results/check',
                   seeds=[23], algorithms=['CTsallis-Action'], dimensions=[[3, 4]],
                   modes=['stationary'], policy_snapshot_interval=1)
        if experiment == 'online':
            cfg['initial_pairs'] = [3, 5]
        return cfg

    def prepare(self, config):
        with patch.object(runner, 'safe_relative', lambda value: self.root / value), redirect_stdout(io.StringIO()):
            return runner.prepare(config)

    def test_resume_replays_prefix_and_matches_uninterrupted_run(self):
        output = self.prepare(self.config())
        task = result_io.read_json(output / 'manifest.json')['tasks'][0]
        full_output = self.root / 'uninterrupted'
        shutil.copytree(output, full_output)
        self.assertEqual(runner.run_one(task, output, True, 1, limit_rounds=4), 'verification_prefix')
        prefix = result_io.read_json(output / 'raw' / (task['task_id'] + '.json'))
        self.assertEqual(prefix['stopped_round'], 4)
        self.assertEqual(task['horizon'], 9)  # A prefix retains the intended learner horizon.
        self.assertEqual(runner.run_one(task, output, True, 1), 'ok')
        self.assertEqual(runner.run_one(task, full_output, True, 1), 'ok')
        resumed = result_io.read_json(output / 'raw' / (task['task_id'] + '.json'))
        self.assertTrue(resumed['prefix_verified'])
        self.assertEqual(resumed['checkpoints']['4'], prefix['checkpoints']['4'])
        with np.load(output / 'traces' / (task['task_id'] + '.npz')) as a, np.load(full_output / 'traces' / (task['task_id'] + '.npz')) as b:
            self.assertEqual(set(a.files), set(b.files))
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])

    def test_failure_is_retained_without_automatic_retry(self):
        output = self.prepare(self.config())
        task = result_io.read_json(output / 'manifest.json')['tasks'][0]
        with patch.object(algorithms, 'instantiate', side_effect=RuntimeError('deliberate test failure')):
            self.assertEqual(runner.run_one(task, output), 'failed')
        record_path = output / 'raw' / (task['task_id'] + '.json')
        before = record_path.read_bytes()
        self.assertIn('deliberate test failure', result_io.read_json(record_path)['reason'])
        with patch.object(algorithms, 'instantiate', side_effect=AssertionError('must not retry')):
            self.assertEqual(runner.run_one(task, output), 'failed')
        self.assertEqual(record_path.read_bytes(), before)

    def test_update_failure_retains_the_incurred_observation(self):
        cfg=self.config();cfg['checkpoints']=[1,9]
        output=self.prepare(cfg)
        task=result_io.read_json(output/'manifest.json')['tasks'][0]
        original=algorithms.instantiate
        def faulty(*args,**kwargs):
            learner=original(*args,**kwargs)
            learner.update=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('update failed'))
            return learner
        with patch.object(algorithms,'instantiate',side_effect=faulty):
            self.assertEqual(runner.run_one(task,output,True,1),'failed')
        record=result_io.read_json(output/'raw'/(task['task_id']+'.json'))
        self.assertEqual(record['stopped_round'],1)
        self.assertEqual(record['checkpoints']['1'],record['final_regret'])
        with np.load(output/'traces'/(task['task_id']+'.npz')) as trace:
            self.assertEqual(len(trace['loss']),1)
            self.assertEqual(trace['cumulative_regret'][-1],record['final_regret'])

    def test_resume_refuses_to_shorten_or_change_a_saved_prefix(self):
        output = self.prepare(self.config())
        task = result_io.read_json(output / 'manifest.json')['tasks'][0]
        self.assertEqual(runner.run_one(task, output, limit_rounds=4), 'verification_prefix')
        path = output / 'raw' / (task['task_id'] + '.json')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            runner.run_one(task, output, limit_rounds=3)
        self.assertEqual(path.read_bytes(), before)
        altered_task = deepcopy(task)
        altered_task['checkpoints'] = [3, 9]
        with self.assertRaises(ValueError):
            runner.run_one(altered_task, output)
        self.assertEqual(path.read_bytes(), before)

    def test_changed_replay_digest_becomes_a_retained_failure(self):
        output = self.prepare(self.config())
        task = result_io.read_json(output / 'manifest.json')['tasks'][0]
        self.assertEqual(runner.run_one(task, output, limit_rounds=4), 'verification_prefix')
        path = output / 'raw' / (task['task_id'] + '.json')
        prior = result_io.read_json(path)
        prior['prefix_digest'] = '0' * 64
        result_io.write_json(path, prior)
        self.assertEqual(runner.run_one(task, output), 'failed')
        failed = result_io.read_json(path)
        self.assertIn('Replayed prefix differs', failed['reason'])
        self.assertFalse(failed['prefix_verified'])
        saved = path.read_bytes()
        self.assertEqual(runner.run_one(task, output), 'failed')
        self.assertEqual(path.read_bytes(), saved)

    def test_inapplicable_method_remains_in_the_plan_without_running(self):
        cfg = self.config()
        cfg.update(algorithms=['CUCB2'], epsilons=[0.])
        output = self.prepare(cfg)
        tasks = result_io.read_json(output / 'manifest.json')['tasks']
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0]['applicable'])
        record = result_io.read_json(output / 'raw' / (tasks[0]['task_id'] + '.json'))
        self.assertEqual(record['status'], 'inapplicable')
        self.assertEqual(record['stopped_round'], 0)
        with patch.object(algorithms, 'instantiate', side_effect=AssertionError('must not instantiate')):
            self.assertEqual(runner.run_one(tasks[0], output), 'inapplicable')

    def test_online_snapshots_precede_observation_and_share_nested_calibration(self):
        cfg = self.config('online')
        cfg['modes'] = ['stationary', 'baseline_switching']
        output = self.prepare(cfg)
        manifest = result_io.read_json(output / 'manifest.json')
        tasks = manifest['tasks']
        self.assertEqual(len(tasks), 6)
        selected = next(t for t in tasks if t['mechanism_mode'] == 'online_3')
        matching = [t for t in tasks if t['mechanism_mode'] == 'online_3']
        self.assertEqual(len({t['calibration_path'] for t in matching}), 1)
        with np.load(output / selected['calibration_path']) as data:
            counts = data['counts'].copy()
        larger = next(t for t in tasks if t['mechanism_mode'] == 'online_5')
        with np.load(output / larger['calibration_path']) as data:
            self.assertTrue(np.all(data['counts'] >= counts))
        self.assertEqual(runner.run_one(selected, output, True, 1), 'ok')
        with np.load(output / 'traces' / (selected['task_id'] + '.npz')) as trace, np.load(output / 'estimates' / (selected['task_id'] + '.npz')) as model:
            np.testing.assert_array_equal(model['model_t'], np.arange(1, 10))
            np.testing.assert_array_equal(model['offline_counts'], counts)
            for i, (a, z) in enumerate(zip(trace['action'], trace['context'])):
                expected = (counts + 1 / counts.shape[0]) / (counts.sum(axis=0) + 1)
                np.testing.assert_allclose(model['M_hat'][i], expected, rtol=0, atol=1e-15)
                counts[z, a] += 1
            np.testing.assert_array_equal(model['final_counts'], counts)
            self.assertEqual(counts.sum(), 3 + 9)
        text = (output / 'manifest.json').read_text()
        self.assertNotIn(str(self.root), text)
        self.assertNotIn(str(result_io.ROOT), text)
        for task in tasks:
            if task['calibration_path']:
                self.assertFalse(Path(task['calibration_path']).is_absolute())
        self.assertTrue(all(not Path(path).is_absolute() for path in manifest['implementation_hashes']))

    def test_prepare_rejects_changed_plan_but_allows_presentation_changes(self):
        cfg = self.config()
        output = self.prepare(cfg)
        cfg['plot']['font_size'] += 1
        self.assertEqual(self.prepare(cfg), output)
        cfg['horizon'] += 1
        with self.assertRaises(ValueError):
            self.prepare(cfg)


if __name__ == '__main__':
    unittest.main()
