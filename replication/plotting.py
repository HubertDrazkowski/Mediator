"""Saved-data-only paper figures, endpoint exports, and transparent missingness.

Rendering never starts learners. An incomplete run can be plotted at any recorded
checkpoint; missing observations, failures and structural N/A remain in exports.
"""
from collections import Counter, defaultdict
from pathlib import Path
import html
import os
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'causal-tsallis-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from .io import read_json, write_csv, write_json, stable_seed
from .algorithms import DISPLAY_NAMES

INF = 'CTsallis-Jensen-selected'
COLORS = dict(zip(
    ['Tsallis-INF', 'CTsallis-Action', 'CTsallis-Context', INF,
     'CTsallis-Jensen-rho0', 'EXP4MF', 'EXP4MF-RV', 'PECUCB', 'CUCB', 'CTS', 'CUCB2'],
    ['#7e8897', '#2474b7', '#339578', '#d04671', '#9a78b7', '#e57b31',
     '#9c4b12', '#d4a85f', '#a18a69', '#20a0b0', '#859552']))
MODE_NAMES = {'stationary': 'Stationary', 'baseline_switching': 'Deterministic switching',
              'deterministic_baseline_switching': 'Deterministic switching',
              'random_baseline_piecewise': 'Piecewise stationary', 'random_piecewise': 'Piecewise stationary'}
OPTIONAL_METADATA = ('orientation', 'n', 'nominal_complexity_ratio', 'nominal_ratio',
                     'source_scenario_id', 'source_stationary_scenario', 'paired_stationary_id', 'switching_source_scenario_id',
                     'rng_scenario_id', 'matched_switching', 'trajectory_seed',
                     'density', 'structure', 'kappa', 'loss_scale')


def uniform_regret(M, vectors, starts, T):
    """Expected pseudo-regret of uniform actions in a synthetic instance."""
    total = 0.
    for i, start in enumerate(starts):
        end = min(T, int(starts[i + 1]) - 1 if i + 1 < len(starts) else T)
        if start > T:
            break
        mu = M.T @ vectors[i]
        total += (end - int(start) + 1) * float(np.mean(mu) - np.min(mu))
    return total


def _finite(value):
    return value is not None and np.isfinite(value)


def _values(rows, metric):
    return [float(r[metric]) for r in rows if _finite(r.get(metric))]


def _write_csv(path, rows, empty_fields=('status',)):
    # io.write_csv intentionally skips empty lists; report files must still exist.
    if rows:
        write_csv(path, rows)
    else:
        Path(path).write_text(','.join(empty_fields) + '\n', encoding='utf-8')


def _statistics(group, metric='cumulative_pseudo_regret'):
    values = _values(group, metric)
    return dict(planned=len(group), available=len(values),
                failed=sum(r['status'] == 'failed' for r in group),
                inapplicable=sum(r['status'] == 'inapplicable' for r in group),
                unavailable=sum(not _finite(r.get(metric)) and r['status'] != 'inapplicable' for r in group),
                status_counts=dict(Counter(r['status'] for r in group)),
                mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                q25=float(np.quantile(values, .25)) if values else None,
                q75=float(np.quantile(values, .75)) if values else None)


def export_results(output):
    """Export all planned endpoints, including pending and inapplicable cells."""
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    cfg = manifest['configuration']
    rows, cache = [], {}
    for sid in manifest['instances']:
        meta = read_json(output / 'instances' / f'{sid}.json')
        with np.load(output / 'instances' / f'{sid}.npz', allow_pickle=False) as arrays:
            # STAR uses the empirical action marginal, NOT a separation projection.
            empirical = next((arrays[k] for k in ('empirical_mu', 'raw_action_means') if k in arrays), None)
            if empirical is None and meta.get('action_means') is not None:
                empirical = np.asarray(meta['action_means'])
            denominators = {int(t): int(t) * float(np.mean(empirical) - np.min(empirical))
                            if empirical is not None else uniform_regret(arrays['M'], arrays['vectors'], arrays['starts'], int(t))
                            for t in cfg['checkpoints']}
        cache[sid] = (meta, denominators)
    for task in manifest['tasks']:
        path = output / 'raw' / f"{task['task_id']}.json"
        record = read_json(path) if path.exists() else dict(status='pending', stopped_round=0)
        meta, denominators = cache[task['scenario_id']]
        for t in task['checkpoints']:
            value = record.get('checkpoints', {}).get(str(t))
            value = float(value) if _finite(value) else None
            if value is not None and value < 0:
                raise ValueError(f"Negative cumulative pseudo-regret: {task['task_id']}, t={t}")
            denom = denominators[int(t)]
            row = dict(experiment=task['experiment'], scenario_id=task['scenario_id'],
                task_id=task['task_id'], algorithm=task['algorithm'],
                algorithm_display=DISPLAY_NAMES.get(task['algorithm'], task['algorithm']).replace('\n', ' '),
                seed=task['seed'], mechanism_mode=task.get('mechanism_mode', 'true'),
                calibration_n=task.get('calibration_n', 0), checkpoint=int(t),
                status=record['status'], stopped_round=record.get('stopped_round', 0),
                checkpoint_available=value is not None,
                n_actions=meta['n_actions'], n_contexts=meta['n_contexts'], epsilon=meta.get('epsilon'),
                mode=meta.get('mode', 'stationary'), cumulative_pseudo_regret=value,
                cumulative_policy_pseudo_regret=record.get('policy_checkpoints', {}).get(str(t)),
                uniform_policy_pseudo_regret=denom,
                normalized_pseudo_regret=value / denom if value is not None and denom > 0 else None,
                action_complexity=meta.get('action_complexity', meta.get('S_A')),
                S_Z_ach=meta.get('context_complexity', meta.get('S_Z_ach')),
                complexity_relation=meta.get('complexity_relation'),
                reason=record.get('reason', task.get('applicability_reason')))
            row.update({k: meta.get(k) for k in OPTIONAL_METADATA})
            rows.append(row)
    _write_csv(output / 'results.csv', rows)
    keys = ['scenario_id', 'algorithm', 'mechanism_mode', 'checkpoint']
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    summary = [dict(zip(keys, key), **_statistics(group)) for key, group in grouped.items()]
    _write_csv(output / 'summary.csv', summary)
    _write_csv(output / 'failures.csv', [r for r in rows if r['status'] in ('failed', 'inapplicable')],
               ('scenario_id', 'algorithm', 'checkpoint', 'status', 'reason'))
    return manifest, rows


def _label(method, rows, multiline=False):
    name = DISPLAY_NAMES.get(method, method).replace('\n', ' ')
    if multiline:
        name = name.replace('CTsallis-', 'CTsallis-\n')
    failed = any(r['algorithm'] == method and r['status'] == 'failed'
                 and not r['checkpoint_available'] for r in rows)
    return name + ('*' if failed else '')


def draw_box(ax, group, position, method, width, metric, tag, size=22, log=False):
    values = _values(group, metric)
    positive = [v for v in values if v > 0] if log else values
    color = COLORS.get(method, '#555555')
    if positive:
        box = ax.boxplot(positive, positions=[position], widths=width, patch_artist=True,
                         showfliers=False, manage_ticks=False,
                         medianprops={'color': color, 'linewidth': 2.8},
                         whiskerprops={'color': color, 'linewidth': 1.7},
                         capprops={'color': color, 'linewidth': 1.7})
        box['boxes'][0].set(facecolor=matplotlib.colors.to_rgba(color, .33), edgecolor=color, linewidth=2)
        jitter = np.random.default_rng(stable_seed('plot_jitter', tag)).uniform(-width * .27, width * .27, len(positive))
        ax.scatter(position + jitter, positive, s=28, color=color, alpha=.77,
                   edgecolor='white', linewidth=.4, zorder=4)
    else:
        note = 'N/A' if group and all(r['status'] == 'inapplicable' for r in group) else 'pending'
        if any(r['status'] == 'failed' for r in group):
            note = 'failed*'
        if values and log:
            note = 'all zero'
        elif group and metric == 'normalized_pseudo_regret' and all(r['checkpoint_available'] for r in group):
            note = 'undefined'
        # Axes-relative labels never determine y limits.
        ax.text(position, .075, note, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=size * .63, color='#666666')
    zeros = len(values) - len(positive)
    if log and zeros and positive:
        ax.text(position, .025, f'{zeros} zero', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=size * .55, color='#666666')
    missing = len(group) - len(values) - sum(r['status'] == 'inapplicable' for r in group)
    if missing and positive:
        ax.text(position, .98, f'{len(values)}/{len(group)}', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=size * .5, color='#555555')
    return values


def style(ax, size, ylabel='Cumulative pseudo-regret'):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size * 1.15, labelpad=14)
    ax.tick_params(axis='both', labelsize=size * .9, pad=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#e6e6e6', alpha=.7)
    ax.spines[['top', 'right']].set_visible(False)


def _set_limits(axes, rows, metric, log=False):
    values = _values(rows, metric)
    if log:
        positives = [v for v in values if v > 0]
        low, high = (min(positives) * .65, max(positives) * 1.5) if positives else (.01, 1.)
        for ax in axes:
            ax.set_yscale('log')
            ax.set_ylim(low, high)
        return
    upper = max(.01, max(values, default=1.)) * 1.06
    ticks = MaxNLocator(nbins=6).tick_values(0, upper)
    ticks = ticks[(ticks >= 0) & (ticks <= upper)]
    for ax in axes:
        ax.set_yticks(ticks)
        ax.set_ylim(0, upper)


def _paired_random(rows):
    """Match graph, method and seed; preserve missing members in the export."""
    stationary = {(r['scenario_id'], r['algorithm'], r['seed']): r for r in rows if r['mode'] == 'stationary'}
    left, right, differences = [], [], []
    for switched in rows:
        if switched['mode'] == 'stationary':
            continue
        sid = next((switched.get(k) for k in ('switching_source_scenario_id', 'paired_stationary_id',
                    'source_scenario_id', 'source_stationary_scenario', 'rng_scenario_id') if switched.get(k)), None)
        if sid is None:
            sid = switched['scenario_id'].removesuffix('__switching')
        original = stationary.get((sid, switched['algorithm'], switched['seed']))
        a = original.get('normalized_pseudo_regret') if original else None
        b = switched.get('normalized_pseudo_regret')
        available = _finite(a) and _finite(b)
        differences.append(dict(stationary_scenario_id=sid, switching_scenario_id=switched['scenario_id'],
            algorithm=switched['algorithm'], seed=switched['seed'], checkpoint=switched['checkpoint'],
            stationary_status=original['status'] if original else 'missing_task',
            switching_status=switched['status'], pair_available=available,
            stationary_regret=a, switching_regret=b, switching_minus_stationary=b-a if available else None))
        if original:
            # Matched figures use complete pairs only. Keep missing/N/A rows as
            # records, with pair-incomplete values cleared in BOTH panels.
            l, r = dict(original), dict(switched)
            if not available:
                l['normalized_pseudo_regret'] = r['normalized_pseudo_regret'] = None
                l['pair_omission'] = r['pair_omission'] = 'incomplete_pair'
            left.append(l)
            right.append(r)
    return left, right, differences


def plot_results(output, plot_options=None):
    output = Path(output)
    manifest, all_rows = export_results(output)
    cfg = manifest['configuration']
    settings = dict(cfg.get('plot', {}))
    settings.update(plot_options or {})
    checkpoint = int(settings.get('checkpoint', cfg['horizon']))
    if checkpoint not in cfg['checkpoints']:
        raise ValueError('Plot checkpoint is not recorded in this experiment')
    rows = [r for r in all_rows if r['checkpoint'] == checkpoint]
    methods = [s['id'] for s in manifest['algorithms']]
    size = float(settings.get('font_size', 28))
    figures, plotting, panel_summary, medians = [], [], [], []
    report = output / 'plots'
    report.mkdir(exist_ok=True)

    def finish(fig, name, description):
        paths = {}
        for ext in settings.get('formats', ['png', 'pdf', 'svg']):
            if ext not in ('png', 'pdf', 'svg'):
                raise ValueError('Unsupported figure format')
            paths[ext] = name + '.' + ext
            metadata = {'Creator': 'Replication scripts'} if ext == 'pdf' else None
            fig.savefig(report / paths[ext], dpi=180, bbox_inches='tight', facecolor='white', metadata=metadata)
        plt.close(fig)
        figures.append(dict(name=name, description=description, **paths))

    def record_panel(selected, name, panel, metric, panel_methods, log=False):
        for method in panel_methods:
            group = [r for r in selected if r['algorithm'] == method]
            panel_summary.append(dict(figure=name, panel=panel, algorithm=method, metric=metric,
                                      environments=len({r['scenario_id'] for r in group}), **_statistics(group, metric)))
            for row in group:
                value = row.get(metric)
                plotted = _finite(value) and (not log or value > 0)
                omission = ('zero_not_representable_on_log_axis' if _finite(value) and log and value == 0
                            else row.get('pair_omission', '') if not plotted else '')
                if not plotted and not omission:
                    omission = row['status'] if not row['checkpoint_available'] else 'normalizer_not_positive'
                plotting.append(dict(row, figure=name, panel=panel, metric=metric,
                                     plot_value=value, plotted=plotted, omission_reason=omission))

    def algorithm_panel(ax, selected, name, panel, panel_methods=None, normalized=False,
                        dash=None, log=False, multiline=False, ylabel=True):
        selected_methods = panel_methods or methods
        metric = 'normalized_pseudo_regret' if normalized else 'cumulative_pseudo_regret'
        for pos, method in enumerate(selected_methods, 1):
            group = [r for r in selected if r['algorithm'] == method]
            values = draw_box(ax, group, pos, method, .60, metric, (name, panel, method), size=size, log=log)
            if method == INF and values and dash:
                median = float(np.median(values))
                if dash == 'full':
                    ax.axhline(median, xmin=0., xmax=1., color=COLORS[INF], linestyle='--', linewidth=2.5, zorder=3)
                else:
                    ax.hlines(median, .5, pos, color=COLORS[INF], linestyle='--', linewidth=2.5, zorder=3)
                medians.append(dict(figure=name, panel=panel, algorithm=INF, median=median,
                                    n=len(values), direction=dash, metric=metric))
        ax.set_xticks(range(1, len(selected_methods) + 1), [_label(m, selected, multiline) for m in selected_methods],
                      rotation=0 if multiline else 47, ha='center' if multiline else 'right', rotation_mode='anchor')
        ax.set_xlim(.5, len(selected_methods) + .5)
        ax.set_xlabel('Algorithm', fontsize=size * 1.05, labelpad=13)
        style(ax, size, ('Cumulative pseudo-regret\n(normalized)' if normalized else 'Cumulative pseudo-regret') if ylabel else '')
        _set_limits([ax], selected, metric, log)
        record_panel(selected, name, panel, metric, selected_methods, log)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': size,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        if cfg['experiment'] == 'e1':
            pairs, regimes = cfg['dimensions'], cfg['modes']
            epsilons = sorted({r['epsilon'] for r in rows})
            for epsilon in epsilons:
                suffix = '' if len(epsilons) == 1 else '_eps' + str(epsilon).replace('.', 'p')
                for focus in (False, True):
                    dims = [settings.get('focus_dimensions', pairs[0])] if focus else pairs
                    focus_modes=settings.get('focus_modes',['stationary','baseline_switching'])
                    plot_regimes=([r for r in regimes if r in focus_modes] or regimes) if focus else regimes
                    selected = [r for r in rows if [r['n_actions'], r['n_contexts']] in dims and r['epsilon'] == epsilon and r['mode'] in plot_regimes]
                    name = ('E1_focus' if focus else 'E1_all') + suffix
                    nr, nc = (1, len(plot_regimes)) if focus else (len(plot_regimes), len(dims))
                    fig, axes = plt.subplots(nr, nc, figsize=(14 * nc, 11 * nr), squeeze=False)
                    for ri, regime in enumerate(plot_regimes):
                        for ci, (A, Z) in enumerate(dims):
                            ax = axes[0, ri] if focus else axes[ri, ci]
                            local = [r for r in selected if r['mode'] == regime and r['n_actions'] == A and r['n_contexts'] == Z]
                            algorithm_panel(ax, local, name, f'{regime}/A{A}/Z{Z}/eps{epsilon}')
                            if not focus:
                                ax.set_xlabel('')
                            if settings.get('show_headers', True):
                                heading = MODE_NAMES.get(regime, regime)
                                if not focus:
                                    heading = fr'$|A|={A},\ |Z|={Z},\ \varepsilon={epsilon}$' + '\n' + heading
                                ax.set_title(heading, fontsize=size * 1.2, pad=20)
                    _set_limits(list(axes.flat), selected, 'cumulative_pseudo_regret')
                    if not focus:
                        fig.supxlabel('Algorithm', fontsize=size * 1.15, y=.025)
                    fig.subplots_adjust(bottom=.30 if focus else .13, top=.9, wspace=.22, hspace=.82)
                    modes_text = ', '.join(MODE_NAMES.get(mode, mode).lower() for mode in plot_regimes)
                    finish(fig, name, 'E1: fixed transition mechanism; ' + modes_text + '. Each dot is one available trajectory seed.')
        elif cfg['experiment'] == 'random':
            stationary = [r for r in rows if r['mode'] == 'stationary']
            switching = [r for r in rows if r['mode'] != 'stationary']
            matched_left, matched_right, pairs = _paired_random(rows)
            _write_csv(output / 'paired_differences.csv', pairs,
                       ('stationary_scenario_id', 'switching_scenario_id', 'algorithm', 'pair_available'))
            specs = [('Random_stationary', [('Stationary', stationary)], 'All stationary environments.')]
            if switching:
                specs += [('Random_stationary_switching', [('Stationary', stationary), ('Deterministic switching', switching)],
                           'All stationary environments and the selected switching cohort; aggregate populations differ.'),
                          ('Random_matched_stationary_switching', [('Stationary', matched_left), ('Deterministic switching', matched_right)],
                           'Matched graph, method and seed; both endpoints must be available for a pair to be plotted.')]
            for name, panels, caption in specs:
                fig, axes = plt.subplots(1, len(panels), figsize=(24 if len(panels) == 1 else 36, 13), squeeze=False)
                for ax, (heading, selected) in zip(axes.flat, panels):
                    algorithm_panel(ax, selected, name, heading, normalized=True, dash='full', ylabel=ax is axes.flat[0])
                    if len(panels) > 1:
                        ax.set_title(heading, fontsize=size * 1.25, pad=22)
                _set_limits(list(axes.flat), [r for _, group in panels for r in group], 'normalized_pseudo_regret')
                fig.subplots_adjust(left=.085, right=.99, bottom=.33, top=.9, wspace=.12)
                finish(fig, name, caption + ' Regret is divided by each instance’s expected uniform-policy regret. Dashed red lines span each panel at the CTsallis-INF median.')
        elif cfg['experiment'] == 'transpose':
            orientations = [o for o in ('vertex', 'edge') if any(str(r.get('orientation', '')).removesuffix('_actions') == o for r in rows)]
            sizes = sorted({int(r['n']) for r in rows if r.get('n') is not None})
            if not orientations or not sizes:
                raise ValueError('Transpose instance metadata requires orientation and n')
            name = f'Transpose_boxplots_t{checkpoint}'
            fig, axes = plt.subplots(len(orientations), len(sizes), figsize=(9 * len(sizes), 7.8 * len(orientations)), squeeze=False)
            for ri, orientation in enumerate(orientations):
                for ci, n in enumerate(sizes):
                    selected = [r for r in rows if str(r.get('orientation', '')).removesuffix('_actions') == orientation and r['n'] == n]
                    ax = axes[ri, ci]
                    algorithm_panel(ax, selected, name, f'{orientation}/n{n}', dash='left', multiline=True)
                    ratio = next((r.get('nominal_complexity_ratio', None) or r.get('nominal_ratio', None) for r in selected
                                  if r.get('nominal_complexity_ratio') is not None or r.get('nominal_ratio') is not None), None)
                    if ratio is None and selected:
                        sa, sz = selected[0]['action_complexity'], selected[0]['S_Z_ach']
                        if _finite(sa) and _finite(sz) and sz > 0:
                            ratio = sa / sz
                    text = f'{ratio:.4g}' if _finite(ratio) else r'\mathrm{N/A}'
                    ax.set_title(r'$S_A/S_Z^{\mathrm{ach}}=' + text + '$', fontsize=size * 1.12, pad=20)
            fig.subplots_adjust(left=.07, right=.985, bottom=.10, top=.90, wspace=.27, hspace=.55)
            finish(fig, name, 'Complete-graph incidence construction in two orientations. Panel headings give the attainable-complexity ratio; each dashed red median extends left from CTsallis-INF.')
        elif cfg['experiment'] == 'online':
            dimensions, regimes = cfg['dimensions'], cfg['modes']
            counts = sorted(set(cfg['initial_pairs']))
            conditions = ['true'] + ['online_' + str(n) for n in counts]
            labels = ['True $M$'] + [f'Online $\\widehat M$\n{n:,}\ninitial pairs' for n in counts]
            epsilons = sorted({r['epsilon'] for r in rows})
            for epsilon in epsilons:
                name = 'Online_mechanism' + ('' if len(epsilons) == 1 else '_eps' + str(epsilon).replace('.', 'p'))
                selected_all = [r for r in rows if r['epsilon'] == epsilon]
                fig, axes = plt.subplots(len(regimes), len(dimensions), figsize=(14 * len(dimensions), 13 * len(regimes)), squeeze=False)
                offsets = np.linspace(-.38, .38, len(methods))
                width = .76 / len(methods) * .86
                for ri, regime in enumerate(regimes):
                    for ci, (A, Z) in enumerate(dimensions):
                        ax = axes[ri, ci]
                        selected = [r for r in selected_all if r['mode'] == regime and r['n_actions'] == A and r['n_contexts'] == Z]
                        for x, condition in enumerate(conditions, 1):
                            group = [r for r in selected if r['mechanism_mode'] == condition]
                            for method, offset in zip(methods, offsets):
                                draw_box(ax, [r for r in group if r['algorithm'] == method], x + offset,
                                         method, width, 'cumulative_pseudo_regret', (name, ri, ci, condition, method), size=size)
                            record_panel(group, name, f'{regime}/A{A}/Z{Z}/{condition}', 'cumulative_pseudo_regret', methods)
                        ax.set_xticks(range(1, len(conditions) + 1), labels)
                        ax.set_xlim(.45, len(conditions) + .55)
                        style(ax, size * 1.2, '')
                        if settings.get('show_headers', True):
                            ax.set_title(fr'$|A|={A},\ |Z|={Z},\ \varepsilon={epsilon}$' + '\n' + MODE_NAMES.get(regime, regime),
                                         fontsize=size * 1.4, pad=22)
                _set_limits(list(axes.flat), selected_all, 'cumulative_pseudo_regret')
                fig.supylabel('Cumulative pseudo-regret', fontsize=size * 1.5, x=.008)
                fig.supxlabel('Mechanism supplied to learner', fontsize=size * 1.4, y=.018)
                handles = [Patch(facecolor=COLORS.get(m, '#555555'), alpha=.45, label=_label(m, selected_all)) for m in methods]
                fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=4, fontsize=size, frameon=False)
                fig.subplots_adjust(left=.075, right=.992, bottom=.18, top=.76, wspace=.32, hspace=.85)
                finish(fig, name, 'True M and online estimates initialized from uniform-action calibration samples; all conditions use the same true environment for evaluation. The shared legend identifies all methods.')
        elif cfg['experiment'] == 'star':
            name = f'STAR_main_log_t{checkpoint}'
            fig, ax = plt.subplots(figsize=(20, 11))
            algorithm_panel(ax, rows, name, 'STAR full mediator', log=True)
            fig.subplots_adjust(left=.10, right=.99, top=.98, bottom=.34)
            finish(fig, name, 'Stationary empirical Project STAR benchmark using the complete five-category mediator. The shared vertical axis is logarithmic; regret uses actual empirical action means. Zero endpoints, if any, are annotated and exported without adding an artificial positive value.')
        else:
            raise ValueError('Unknown plotting experiment: ' + cfg['experiment'])

    _write_csv(output / 'plotting_data.csv', plotting)
    _write_csv(output / 'panel_summary.csv', panel_summary)
    _write_csv(output / 'dashed_medians.csv', medians, ('figure', 'panel', 'algorithm', 'median', 'n'))
    counts = dict(Counter(r['status'] for r in rows))
    available = sum(r['checkpoint_available'] for r in rows)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Replication results</title>',
            '<style>body{font:18px/1.6 system-ui;margin:30px auto;max-width:1750px;padding:0 24px}img{width:100%;height:auto}a{color:#176c9c}figure{margin:35px 0}figcaption{font-size:16px}</style>',
            f'<h1>{html.escape(cfg["experiment"])}: T={checkpoint:,}</h1>',
            f'<p>{available}/{len(rows)} checkpoints available. Status counts: {html.escape(str(counts))}.</p>',
            '<p><a href="results.csv">All checkpoint results</a> · <a href="summary.csv">Per-instance summary</a> · '
            '<a href="panel_summary.csv">Panel counts and summary</a> · <a href="plotting_data.csv">Exact plotting data</a> · '
            '<a href="failures.csv">Failures and N/A</a> · <a href="manifest.json">Parameters and provenance</a></p>',
            '<p>Boxes show quartiles and the median; whiskers extend to observations within 1.5 IQR. '
            'All available finite endpoints are shown individually. Missing runs are never imputed. '
            '* denotes a failed missing endpoint; N/A denotes a structural applicability restriction. '
            'Annotations are positioned in axes coordinates and do not inflate the data limits.</p>']
    if cfg['experiment'] == 'random':
        page.append('<p>The full stationary and switching panels can contain different environment cohorts. '
                    'The separate matched comparison retains graph–method–seed pairs with both checkpoints available. '
                    'Normalization is by expected uniform-action regret on each fixed instance. '
                    '<a href="paired_differences.csv">Matched differences and missing-pair records</a> · '
                    '<a href="dashed_medians.csv">Dashed median values</a>.</p>')
    if cfg['experiment'] == 'online':
        page.append('<p>Initial calibration uses uniformly sampled actions and mediator draws conditional on the true mechanism. '
                    'Estimates update from online observations. Tsallis-INF ignores the supplied mechanism; its repetitions '
                    'across conditions are not independent trajectory replicates.</p>')
    if cfg['experiment'] == 'star':
        page.append('<p>Each round resamples a retained Project STAR record within the selected treatment arm. '
                    'Its mediator and outcome are observed jointly. This empirical replay permits incomplete mediation; '
                    'pseudo-regret is evaluated against actual empirical arm means, not means reconstructed from mediator averages. '
                    'Seed variability is simulation variability conditional on the empirical cohort.</p>')
    for figure in figures:
        links = ' · '.join(f'<a href="plots/{figure[e]}">{e.upper()}</a>' for e in ('png', 'pdf', 'svg') if e in figure)
        page.append('<figure>')
        if 'png' in figure:
            page.append(f'<img src="plots/{figure["png"]}" alt="{html.escape(figure["description"])}">')
        page.append(f'<figcaption>{html.escape(figure["description"])} {links}</figcaption></figure>')
    (output / 'REPORT.html').write_text('\n'.join(page + ['</html>']), encoding='utf-8')
    write_json(output / 'plot_validation.json', dict(checkpoint=checkpoint, planned=len(rows), available=available,
        status_counts=counts, plotted_points=sum(r['plotted'] for r in plotting), figures=figures,
        saved_data_only=True, experiment_runs_started=0,
        log_zero_points_not_replaced=sum(r['omission_reason'] == 'zero_not_representable_on_log_axis' for r in plotting)))
    print(f'Saved report, CSVs and plots in {output.name}/', flush=True)
