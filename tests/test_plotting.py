"""Paper-panel selection checks using marked test fixtures; no learner runs."""
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from replication import plotting
from matplotlib.figure import Figure


class PaperPanelSelectionTests(unittest.TestCase):
    def test_e1_focus_excludes_piecewise_but_full_plot_retains_it(self):
        methods = [m for m in plotting.COLORS if m != 'EXP4MF-RV']
        regimes = ['stationary', 'baseline_switching', 'random_baseline_piecewise']
        config = dict(experiment='e1', horizon=30000, checkpoints=[30000],
                      dimensions=[[80, 30]], modes=regimes,
                      plot=dict(checkpoint=30000, focus_dimensions=[80, 30],
                                formats=['png'], font_size=20, show_headers=True))
        # Deterministic fixture values test membership/counting, not performance.
        rows = [dict(scenario_id='fixture_' + regime, algorithm=method,
                     seed=seed, mode=regime, epsilon=.3, n_actions=80, n_contexts=30,
                     checkpoint=30000, checkpoint_available=True, status='ok',
                     cumulative_pseudo_regret=float(10 + position + seed - 8201))
                for regime in regimes for position, method in enumerate(methods)
                for seed in range(8201, 8231)]
        manifest = dict(configuration=config, algorithms=[dict(id=m) for m in methods])
        snapshots = {}

        def capture(figure, path, **kwargs):
            snapshots[Path(path).stem] = dict(
                panels=len(figure.axes),
                boxes=sum(len(ax.patches) for ax in figure.axes),
                points=sum(len(collection.get_offsets()) for ax in figure.axes for collection in ax.collections),
                titles=[ax.get_title() for ax in figure.axes])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(plotting, 'export_results', return_value=(manifest, rows)), \
                 patch.object(Figure, 'savefig', autospec=True, side_effect=capture), \
                 redirect_stdout(io.StringIO()):
                plotting.plot_results(output)
            with (output / 'plotting_data.csv').open() as handle:
                exported = list(csv.DictReader(handle))
            focus = [r for r in exported if r['figure'] == 'E1_focus']
            full = [r for r in exported if r['figure'] == 'E1_all']
            self.assertEqual({r['mode'] for r in focus}, set(regimes[:2]))
            self.assertEqual({r['mode'] for r in full}, set(regimes))
            self.assertEqual(len(focus), 600)
            self.assertEqual(len(full), 900)
        self.assertEqual(snapshots['E1_focus'], dict(panels=2, boxes=20, points=600,
            titles=['Stationary', 'Deterministic switching']))
        self.assertEqual(snapshots['E1_all']['panels'], 3)
        self.assertEqual(snapshots['E1_all']['boxes'], 30)
        self.assertEqual(snapshots['E1_all']['points'], 900)


if __name__ == '__main__':
    unittest.main()
