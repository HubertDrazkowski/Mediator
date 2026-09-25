"""Standalone generators for the synthetic paper experiment families.

An instance is a dictionary with scenario_id, M, vectors, starts,
baseline_groups and metadata. M has shape (Z,A); starts are one-based.
No function in this module imports or runs a learner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math

import numpy as np

RANDOM_DEFAULTS = dict(seed=2026091802, horizon=30000, n_graphs=50,
    losses_per_graph=1, dimensions=[(a, z) for a in (10, 30, 80) for z in (10, 30, 80)],
    densities=[.05, .2, .4], structures=[0., .5, 1.], kappas=[.3, 1., 10.],
    lambdas=[.03, .1, .3], joint_diagnostics=True)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _tag(value):
    return str(value).replace('.', 'p')


def _words(*parts):
    digest = hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()
    return list(map(int, np.frombuffer(digest[:16], dtype='<u4')))


def _rng(seed, *parts):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *_words(*parts)]))


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(value)


def geometric_starts(horizon, L0=50, gamma=1.6):
    """Exact historical schedule: ceil(length), then length *= gamma.

    Use this recurrence, not ceil(L0 * gamma**k): binary floating-point
    exponentiation can round an exact integer upward (notably 128 at k=2).
    The final segment is truncated to the horizon.
    """
    horizon, L0 = _positive_integer(horizon, 'horizon'), _positive_integer(L0, 'L0')
    if not np.isfinite(gamma) or gamma <= 1:
        raise ValueError('gamma must be finite and greater than one')
    starts, start, length = [], 1, float(L0)
    while start <= horizon:
        starts.append(start)
        start += max(1, math.ceil(length))
        length = min(float(horizon), length * gamma)
    return np.asarray(starts, dtype=np.int32)


def complexity_relation(S_A, lower, upper, rtol=1e-10):
    """Certify a comparison against bounds on joint attainable complexity."""
    if lower is None or upper is None:
        return 'unresolved_by_bounds'
    if S_A > upper + rtol * max(1., abs(S_A), abs(upper)):
        return 'SA_gt_SZ'
    if S_A < lower - rtol * max(1., abs(S_A), abs(lower)):
        return 'SA_lt_SZ'
    if lower == upper and abs(S_A - upper) <= rtol * max(1., abs(S_A), abs(upper)):
        return 'SA_eq_SZ'
    return 'unresolved_by_bounds'


def _joint_complexity(M, gaps, outside, tie_atol):
    """min sum(1/gamma) subject to M_U.T @ gamma <= Delta_A, gamma>0.

    Independent one-coordinate opening margins only scale the optimization;
    their reciprocal sum is NOT the reported attainable mediator complexity.
    A feasible primal value and a dual lower bound certify its approximation.
    """
    from scipy.optimize import minimize, nnls

    reachable = np.any(M > 0, axis=1)
    opening = np.flatnonzero(outside & reachable)
    result = dict(opening_contexts=opening.tolist(), unreachable_contexts=np.flatnonzero(~reachable).tolist())
    if not len(opening):
        return dict(result, context_complexity=0., context_complexity_lower_bound=0.,
                    context_complexity_upper_bound=0., complexity_relative_gap=0.,
                    complexity_status='empty_opening_support', context_substitution_margins=[])
    B = M[opening]
    optimal = gaps <= tie_atol
    if np.any(B[:, optimal] > 0):
        return dict(result, context_complexity=None, context_complexity_lower_bound=None,
                    context_complexity_upper_bound=None, complexity_relative_gap=None,
                    complexity_status='infinite_canonical_support_with_different_tied_optima',
                    context_substitution_margins=None)
    B, d = B[:, ~optimal], gaps[~optimal]
    with np.errstate(divide='ignore', invalid='ignore'):
        coordinate_bounds = np.min(np.where(B > 0, d[None, :] / B, np.inf), axis=1)
    if not np.all(np.isfinite(coordinate_bounds)) or np.any(coordinate_bounds <= 0):
        raise ValueError('Invalid opening-margin bounds')
    constraints = (B.T * coordinate_bounds[None, :]) / d[:, None]
    weights = 1 / coordinate_bounds
    weights /= weights.sum()
    initial = np.full(len(opening), .5 / max(1., float(constraints.sum(axis=1).max())))
    optimized = minimize(lambda u: float(np.sum(weights / u)), initial,
        jac=lambda u: -weights / u**2, bounds=[(1e-12, 1.)] * len(opening),
        constraints={'type': 'ineq', 'fun': lambda u: 1 - constraints @ u,
                     'jac': lambda u: -constraints}, method='SLSQP',
        options={'ftol': 1e-10, 'maxiter': 1500, 'disp': False})
    u = np.asarray(optimized.x)
    if not np.isfinite(u).all() or np.any(u <= 0):
        raise FloatingPointError('Joint-margin optimizer did not return positive finite coordinates')
    u /= max(1., float(np.max(constraints @ u))) * (1 + 1e-12)
    gamma = coordinate_bounds * u
    primal = float(np.sum(1 / gamma))
    slack = d - B.T @ gamma
    active = np.flatnonzero(slack <= 1e-5 * np.maximum(d, 1e-12))
    if not len(active):
        active = np.arange(len(d))
    target = 1 / gamma**2
    try:
        multipliers, _ = nnls(B[:, active], target / target.max(), maxiter=max(1000, 20 * len(active)))
    except RuntimeError:
        multipliers = np.ones(len(active))
    if not np.any(multipliers > 0):
        multipliers = np.ones(len(active))
    dual = float(np.sqrt(B[:, active] @ multipliers).sum()**2 / (d[active] @ multipliers))
    relative_gap = max(0., (primal - dual) / max(primal, 1e-300))
    return dict(result, context_complexity=primal, context_complexity_lower_bound=dual,
        context_complexity_upper_bound=primal, complexity_relative_gap=relative_gap,
        complexity_status='numerical_primal_dual_certified' if optimized.success and relative_gap < 1e-5 else 'numerical_bound_only',
        context_substitution_margins=gamma.tolist(), independent_opening_margins=coordinate_bounds.tolist(),
        solver_success=bool(optimized.success), solver_message=str(optimized.message))


def complexity_diagnostics(M, g, tie_atol=1e-12):
    """S_A and S_Z^ach for a fixed M and mean vector, including tie handling.

    U is outside the support of the lowest-index optimal action, not simply
    all mediators with loss exceeding min(g). A separate union-support value
    documents tied optima. Unreachable rows are excluded. The original paper sample's
    tied optimal actions all have the same support.
    """
    M, g = np.asarray(M, dtype=float), np.asarray(g, dtype=float)
    if M.ndim != 2 or g.shape != (M.shape[0],) or not np.isfinite(M).all() or not np.isfinite(g).all():
        raise ValueError('Require a finite Z-by-A matrix and Z-vector of means')
    if np.any(M < 0) or not np.allclose(M.sum(axis=0), 1., atol=1e-12, rtol=0):
        raise ValueError('M must be column stochastic')
    mu = M.T @ g
    gaps = mu - mu.min()
    optima = np.flatnonzero(gaps <= tie_atol)
    canonical = int(optima[0])
    result = dict(action_means=mu.tolist(), action_gaps=gaps.tolist(),
                  action_complexity=float(np.sum(1 / gaps[gaps > tie_atol])),
                  optimal_actions=optima.tolist(), canonical_optimal_action=canonical, tie_tolerance=tie_atol)
    result.update(_joint_complexity(M, gaps, M[:, canonical] == 0, tie_atol))
    union = np.any(M[:, optima] > 0, axis=1)
    union_result = (_joint_complexity(M, gaps, ~union, tie_atol)
                    if np.any(union != (M[:, canonical] > 0)) else result)
    result['union_support_context_complexity'] = union_result['context_complexity']
    result['union_support_complexity_status'] = union_result['complexity_status']
    result['S_A'], result['S_Z_ach'] = result['action_complexity'], result['context_complexity']
    result['complexity_relation'] = complexity_relation(result['action_complexity'],
        result['context_complexity_lower_bound'], result['context_complexity_upper_bound'])
    result['complexity_relation_certified_by_bounds'] = result['complexity_relation'] != 'unresolved_by_bounds'
    return result


def _diagnostics_safely(M, g):
    try:
        return complexity_diagnostics(M, g)
    except Exception as exc:
        mu = M.T @ g
        gaps = mu - mu.min()
        return dict(action_means=mu.tolist(), action_gaps=gaps.tolist(),
                    action_complexity=float(np.sum(1 / gaps[gaps > 1e-12])),
                    context_complexity=None, context_complexity_lower_bound=None,
                    context_complexity_upper_bound=None, complexity_status='numerical_diagnostic_failed',
                    complexity_relation='unresolved_by_bounds', complexity_relation_certified_by_bounds=False,
                    diagnostic_error_type=type(exc).__name__, diagnostic_error=str(exc))


def _finish(sid, M, vectors, starts, metadata, horizon, diagnostics=True):
    M, vectors = np.asarray(M, dtype=float), np.asarray(vectors, dtype=float)
    starts = np.asarray(starts, dtype=np.int32)
    meta = deepcopy(metadata)
    meta.update(scenario_id=sid, n_actions=M.shape[1], n_contexts=M.shape[0], horizon=int(horizon),
                schedule_starts=starts.tolist(), matrix_sha256=array_sha256(M), losses_sha256=array_sha256(vectors),
                mechanism_fixed_over_time=True, no_outcome_selection=True)
    cache, phases = {}, []
    if diagnostics:
        for i, (start, g) in enumerate(zip(starts, vectors)):
            key = array_sha256(g)
            if key not in cache:
                cache[key] = _diagnostics_safely(M, g)
            end = int(starts[i + 1] - 1) if i + 1 < len(starts) else horizon
            phases.append(dict(cache[key], phase_index=i, start_round=int(start), end_round=end,
                               length=end - int(start) + 1))
        meta['phase_diagnostics'] = phases
        meta.update({key: phases[0][key] for key in ('action_complexity', 'context_complexity', 'complexity_relation')})
        meta['S_A'], meta['S_Z_ach'] = meta['action_complexity'], meta['context_complexity']
        meta['initial_complexity_relation'] = meta['complexity_relation']
    else:
        meta.update(action_complexity=None, context_complexity=None, S_A=None, S_Z_ach=None,
                    complexity_relation='not_computed', initial_complexity_relation='not_computed')
    meta['uniform_regret'] = {}
    for endpoint in sorted({min(5000, horizon), min(30000, horizon), horizon}):
        value = 0.
        for i, g in enumerate(vectors):
            end = min(endpoint, int(starts[i + 1] - 1) if i + 1 < len(starts) else endpoint)
            length = max(0, end - int(starts[i]) + 1)
            mu = M.T @ g
            value += length * float((mu - mu.min()).mean())
        meta['uniform_regret'][str(endpoint)] = value
    return dict(scenario_id=sid, M=M, vectors=vectors, starts=starts,
                baseline_groups=np.arange(M.shape[0], dtype=int), metadata=meta)


def generate_e1(A, Z, epsilon=.3, horizon=30000, delta=.1, L0=50, gamma=1.6):
    """Return stationary and baseline-switching versions of the same E1 M.

    M[z,a]=(1-epsilon)1{z=a mod Z}+epsilon/Z. Baseline switches preserve
    action gaps. A>Z retains repeated columns; A<Z has background-only rows.
    """
    A, Z, horizon = (_positive_integer(v, name) for v, name in ((A, 'A'), (Z, 'Z'), (horizon, 'horizon')))
    if min(A, Z) < 2 or not 0 <= epsilon < 1 or not 0 < delta <= 1:
        raise ValueError('Require A,Z>=2, 0<=epsilon<1 and 0<delta<=1')
    dominant = np.arange(A) % Z
    M = np.full((Z, A), epsilon / Z)
    M[dominant, np.arange(A)] += 1 - epsilon
    stationary = np.full(Z, .5 + delta / 2)
    stationary[0] = .5 - delta / 2
    low = np.full(Z, delta)
    low[0] = 0.
    switch_starts = geometric_starts(horizon, L0, gamma)
    graph = f'e1random30_e1_near_identity_a{A}_z{Z}_eps{_tag(epsilon)}'
    results = []
    for mode in ('stationary', 'baseline_switching'):
        starts = np.array([1], dtype=np.int32) if mode == 'stationary' else switch_starts
        vectors = np.array([stationary]) if mode == 'stationary' else np.array(
            [low if i % 2 == 0 else low + 1 - delta for i in range(len(starts))])
        sid = f'{graph}_{mode}_i000'
        metadata = dict(family='E1', experiment='E1', mode=mode, epsilon=float(epsilon),
            context_gap=float(delta), action_gap=float((1 - epsilon) * delta),
            graph_id=graph + '_i000', loss_id=f'd{_tag(delta)}_{mode}',
            environment_seed=2026091802, dominant_context=dominant.tolist(),
            fixed_optimal_actions=np.flatnonzero(dominant == 0).tolist(),
            baseline_only_contexts=list(range(A, Z)) if A < Z else [],
            unique_columns=min(A, Z), initial_segment_length=int(L0), segment_growth=float(gamma),
            segment_rounding='Iterative float length: start=1; add ceil(length); length=min(horizon,length*gamma).',
            temporal_change='none' if mode == 'stationary' else 'common_baseline_only')
        results.append(_finish(sid, M, vectors, starts, metadata, horizon))
    return results


def _balanced_values(seed, namespace, levels, count):
    values = []
    rng = _rng(seed, 'e1random30_assignment', namespace)
    while len(values) < count:
        order = rng.permutation(len(levels))
        values.extend(levels[i] for i in order[:count - len(values)])
    rng.shuffle(values)
    return values


def _random_graph(seed, number, A, Z, d, r, kappa):
    if min(A, Z) < 2 or not 0 < d or not 0 <= r <= 1 or d * (1 + r) > 1 or kappa <= 0:
        raise ValueError('Invalid dimensions, density, structure or Dirichlet concentration')
    rng = _rng(seed, 'graph', number, A, Z)
    action_blocks, context_blocks = np.arange(A) % 2, np.arange(Z) % 2
    rng.shuffle(action_blocks)
    rng.shuffle(context_blocks)
    probabilities = np.where(context_blocks[:, None] == action_blocks[None, :], d * (1 + r), d * (1 - r))
    support = rng.random((Z, A)) < probabilities
    repairs = []
    for a in np.flatnonzero(~np.any(support, axis=0)):
        z = int(rng.choice(np.flatnonzero(probabilities[:, a] > 0)))
        support[z, a] = True
        repairs.append({'pass': 'action', 'action': int(a), 'context': z})
    for z in np.flatnonzero(~np.any(support, axis=1)):
        a = int(rng.choice(np.flatnonzero(probabilities[z] > 0)))
        support[z, a] = True
        repairs.append({'pass': 'context', 'action': a, 'context': int(z)})
    M = np.zeros((Z, A))
    rng = _rng(seed, 'weights', number, A, Z)
    for a in range(A):
        chosen = np.flatnonzero(support[:, a])
        weights = rng.dirichlet(np.full(len(chosen), kappa))
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise FloatingPointError('Invalid Dirichlet sample; no silent resampling')
        M[chosen, a] = weights
    metadata = dict(graph_number=number, d=d, r=r, kappa=kappa, p_same=d * (1 + r), p_different=d * (1 - r),
        action_blocks=action_blocks.tolist(), context_blocks=context_blocks.tolist(), repairs=repairs,
        repair_rule='Isolated actions first, then isolated mediators; one uniformly chosen edge among positive-probability neighbors.',
        repair_changes_Bernoulli_distribution=True, realized_edge_count=int(support.sum()),
        graph_seed_sequence=[int(seed), *_words('graph', number, A, Z)],
        weight_seed_sequence=[int(seed), *_words('weights', number, A, Z)],
        weight_distribution='Independent symmetric Dirichlet(kappa,...,kappa), with kappa per supported edge.')
    return M, metadata


def generate_random(config=None):
    """Sample the declared balanced two-community Bernoulli/Dirichlet family.

    Accept keys in RANDOM_DEFAULTS. Marginal factor-level counts differ by at
    most one; this is not a full factorial. Changing n_graphs changes the whole
    balanced assignment; use the original n_graphs=50 and seed to regenerate
    the original sample.
    All sampled environments are retained, independent of algorithm outcomes.
    """
    options = deepcopy(RANDOM_DEFAULTS)
    if config:
        unknown = set(config) - set(options)
        if unknown:
            raise ValueError('Unknown random-generation controls: ' + ', '.join(sorted(unknown)))
        options.update(config)
    seed, horizon = int(options['seed']), _positive_integer(options['horizon'], 'horizon')
    graphs = _positive_integer(options['n_graphs'], 'n_graphs')
    losses = _positive_integer(options['losses_per_graph'], 'losses_per_graph')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    levels = {key: tuple(options[name]) for key, name in
              (('dimensions', 'dimensions'), ('density', 'densities'), ('structure', 'structures'),
               ('kappa', 'kappas'), ('lambda', 'lambdas'))}
    if any(not values for values in levels.values()) or any(not 0 < v < .5 for v in levels['lambda']):
        raise ValueError('Every factor needs levels and lambda must lie in (0,.5)')
    assigned = {name: _balanced_values(seed, name, values, graphs * losses if name == 'lambda' else graphs)
                for name, values in levels.items()}
    results = []
    for number, (A, Z) in enumerate(assigned['dimensions']):
        d, r, kappa = (float(assigned[name][number]) for name in ('density', 'structure', 'kappa'))
        M, graphmeta = _random_graph(seed, number, A, Z, d, r, kappa)
        graph = f'e1random30_random_seed{seed}_g{number:03d}_a{A}_z{Z}_d{_tag(d)}_r{_tag(r)}_k{_tag(kappa)}'
        for loss_number in range(losses):
            parts = ('e1random30_loss', number, A, Z, loss_number)
            x = _rng(seed, *parts).uniform(-1., 1., Z)
            strength = float(assigned['lambda'][number * losses + loss_number])
            g = .5 + strength * x
            loss_id = f'{graph}_loss{loss_number:02d}'
            sid = f'{loss_id}_lambda{_tag(strength)}'
            metadata = dict(graphmeta, family='Random', experiment='E6', mode='stationary', epsilon=None,
                environment_seed=seed, graph_id=graph, loss_id=loss_id, lambda_scale=strength,
                unscaled_context_losses=x.tolist(), loss_seed_sequence=[seed, *_words(*parts)],
                n_graphs=graphs, losses_per_graph=losses, loss_number=loss_number,
                parameter_assignment='Independent randomized balanced marginal cycles; level counts differ by at most one; not factorial.',
                loss_sampling='x_z iid Uniform[-1,1], independent of M; g_z=.5+lambda*x_z.', temporal_change='none')
            results.append(_finish(sid, M, [g], [1], metadata, horizon, options['joint_diagnostics']))
    return results


def generate_e1_piecewise(A, Z, trajectory_seed, epsilon=.3, horizon=30000,
                         delta=.1, L0=50, gamma=1.6, beta_low=-.45, beta_high=.45):
    """E1 with a seed-specific, random piecewise common loss baseline.

    Segment offsets are iid Uniform(-.45,.45), independent of learner RNG.
    The same trajectory seed gives the same offsets across dimensions and
    algorithms. The paper uses seeds 8201--8230 and alpha_s=1. With delta=.1,
    clipping is inactive; every optimum and gap is preserved.
    """
    seed = _nonnegative_integer(trajectory_seed, 'trajectory_seed')
    if not np.isfinite(beta_low) or not np.isfinite(beta_high) or beta_low >= beta_high:
        raise ValueError('Require finite beta_low < beta_high')
    stationary = generate_e1(A, Z, epsilon, horizon, delta, L0, gamma)[0]
    g0, M = stationary['vectors'][0], stationary['M']
    starts = geometric_starts(horizon, L0, gamma)
    # Keep default json separators: this is a distinct historical RNG namespace.
    parts = ('e1_piecewise_uniform_baseline_v1', seed)
    schedule_seed = int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:16], 'little')
    betas = np.random.default_rng(schedule_seed).uniform(beta_low, beta_high, len(starts))
    raw = .5 + betas[:, None] + (g0[None, :] - .5)
    vectors = np.clip(raw, 0., 1.)
    sid = f'e1_random_baseline_a{A}_z{Z}_eps{_tag(epsilon)}_seed{seed}'
    metadata = dict(stationary['metadata'])
    metadata.update(mode='random_baseline_piecewise', alpha=1.,
        beta_min=float(beta_low), beta_max=float(beta_high), beta_values=betas.tolist(), g0=g0.tolist(),
        schedule_seed=schedule_seed, replicate_seed=seed, trajectory_seed=seed,
        source_stationary_scenario=stationary['scenario_id'],
        source_switching_scenario=stationary['scenario_id'].replace('stationary', 'baseline_switching'),
        first_segment=int(L0), growth=float(gamma),
        loss_formula='clip(0.5+beta_s+alpha_s*(g0-0.5),0,1)',
        clipping_active=bool(np.any(raw != vectors)),
        temporal_change='random_common_baseline_only', learner_reset_at_switch=False,
        randomness='iid segment offsets per seed shared across methods and dimensions; independent trajectory RNG')
    result = _finish(sid, M, vectors, starts, metadata, horizon)
    result['trajectory_seed'] = seed
    return result


def generate_transpose(n, orientation, horizon=50000, baseline=.35,
                       minimum_action_gap=.02, environment_seed=2026092307):
    """Transpose two orientations of the complete-graph incidence matrix.

    For vertex actions, the optimal vertex's incident edge-mediators have low
    loss. For edge actions, the optimal edge's two vertex-mediators have low
    loss. Loss placement is uniform and independent of any learner outcome.
    The paper evaluates n=8,12,16 in both orientations, with exact analytic
    certificates for the action and jointly attainable mediator complexities.
    """
    n, horizon = _positive_integer(n, 'n'), _positive_integer(horizon, 'horizon')
    if n < 3 or orientation not in ('vertex', 'edge'):
        raise ValueError("Require n>=3 and orientation 'vertex' or 'edge'")
    if not np.isfinite(minimum_action_gap) or minimum_action_gap <= 0:
        raise ValueError('minimum_action_gap must be positive and finite')
    master = int(environment_seed)
    if master < 0 or master != environment_seed:
        raise ValueError('environment_seed must be a nonnegative integer')
    edges = np.asarray(list(combinations(range(n), 2)), dtype=int)
    incidence = np.zeros((len(edges), n))
    incidence[np.arange(len(edges))[:, None], edges] = 1.
    parts = (master, 'transpose_complete_graph_v1', n, orientation)
    seed = int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:16], 'little')
    rng = np.random.default_rng(seed)
    if orientation == 'vertex':
        M = incidence / (n - 1)
        optimal = int(rng.integers(n))
        low = incidence[:, optimal] > 0
        mediator_gap = minimum_action_gap * (n - 1) / (n - 2)
        sa = (n - 1) / minimum_action_gap
        sz = ((n - 1) * (n - 2) / 2) / mediator_gap
        dual = np.full(n, (n - 1) / (2 * mediator_gap**2))
        optimum = dict(optimal_vertex=optimal, optimal_edge=None)
        action_nodes, mediator_nodes = list(range(n)), edges.tolist()
        formula_sa, formula_sz = '(n-1)/D', '((n-1)*(n-2)/2)/delta_Z'
        formula_ratio = '2*(n-1)/(n-2)**2'
    else:
        M = incidence.T / 2
        optimal = int(rng.integers(len(edges)))
        low = np.zeros(n, dtype=bool)
        low[edges[optimal]] = True
        mediator_gap = 2 * minimum_action_gap
        sa = (n - 2) * (n + 5) / (2 * mediator_gap)
        sz = (n - 2) / mediator_gap
        dual = np.full(len(edges), 2 / ((n - 1) * mediator_gap**2))
        optimum = dict(optimal_vertex=None, optimal_edge=edges[optimal].tolist())
        action_nodes, mediator_nodes = edges.tolist(), list(range(n))
        formula_sa, formula_sz = '(n-2)*(n+5)/(2*delta_Z)', '(n-2)/delta_Z'
        formula_ratio = '(n+5)/2'
    if not np.isfinite(baseline) or baseline < 0 or baseline + mediator_gap > 1:
        raise ValueError('baseline and baseline+mediator_gap must lie in [0,1]')
    dual[optimal] = 0.
    means = np.full(M.shape[0], baseline + mediator_gap)
    means[low] = baseline
    opening = np.flatnonzero(~low)
    mu = M.T @ means
    gaps = mu - mu.min()
    sid = f'transpose_{orientation}_n{n}'
    analytic = dict(action_complexity=sa, context_complexity=sz, S_A=sa, S_Z_ach=sz,
        context_complexity_lower_bound=sz, context_complexity_upper_bound=sz,
        complexity_relative_gap=0., complexity_ratio=sa / sz, target_ratio=sa / sz,
        complexity_relation='SA_lt_SZ' if sa < sz else ('SA_gt_SZ' if sa > sz else 'SA_eq_SZ'),
        complexity_status='analytic_primal_dual_certified',
        complexity_relation_certified_by_bounds=True,
        complexity_certificate='Feasible joint margins and matching nonnegative dual action weights',
        action_complexity_formula=formula_sa, context_complexity_formula=formula_sz,
        complexity_ratio_formula=formula_ratio, opening_contexts=opening.tolist(),
        context_substitution_margins=np.full(len(opening), mediator_gap).tolist(),
        dual_action_weights=dual.tolist(), action_means=mu.tolist(), action_gaps=gaps.tolist(),
        optimal_actions=[optimal], canonical_optimal_action=optimal,
        union_support_context_complexity=sz,
        union_support_complexity_status='analytic_primal_dual_certified')
    metadata = dict(family='TransposeCompleteGraph', experiment='TransposeCompleteGraph',
        graph_id=sid, loss_id=sid + '_loss', orientation=orientation, n=n,
        mode='stationary', epsilon=0., A=M.shape[1], Z=M.shape[0],
        base_mean=baseline, minimum_action_gap=minimum_action_gap,
        mediator_gap=mediator_gap, delta_Z=mediator_gap,
        fixed_optimal_actions=[optimal], unique_optimal_action=True, optimal_action=optimal,
        low_loss_contexts=np.flatnonzero(low).tolist(), equal_column_entropy=True,
        distinct_rows=True, distinct_columns=True,
        graph_generation='Complete undirected graph; lexicographic unordered edges; deterministic incidence matrix',
        action_nodes=action_nodes, mediator_nodes=mediator_nodes,
        environment_master_seed=master, environment_seed=seed,
        environment_seed_derivation="little-endian first 16 SHA256 bytes of json.dumps((master_seed, 'transpose_complete_graph_v1', n, orientation))",
        loss_placement_sampling='One uniform action index using default_rng(environment_seed).integers(A)',
        no_heterogeneity=True, temporal_change='none', **optimum)
    result = _finish(sid, M, [means], [1], metadata, horizon, diagnostics=False)
    result['metadata'].update(analytic)
    result['metadata']['initial_complexity_relation'] = analytic['complexity_relation']
    result['metadata']['phase_diagnostics'] = [dict(analytic, phase_index=0,
        start_round=1, end_round=horizon, length=horizon)]
    return result


def generate_random_paper(horizon=30000, joint_diagnostics=True, config=None):
    """Reproduce the original 50 plus independent 20 stationary environments.

    The balanced generator is not prefix-stable in its sample count. Sampling
    70 graphs in one batch would replace the historical sample. Instead this
    uses seeds 2026091802 and 2026092301 for batches of 50 and 20 respectively.
    Optional config keys are RANDOM_DEFAULTS, extension_seed and
    extension_n_graphs; changing them defines a new declared benchmark.
    """
    options = dict(config or {})
    extension_seed = _nonnegative_integer(options.pop('extension_seed', 2026092301), 'extension_seed')
    extension_count = _nonnegative_integer(options.pop('extension_n_graphs', 20), 'extension_n_graphs')
    options.setdefault('seed', RANDOM_DEFAULTS['seed'])
    options.setdefault('n_graphs', 50)
    options.update(horizon=horizon, joint_diagnostics=joint_diagnostics)
    original = generate_random(options)
    extension_config = dict(options, seed=extension_seed, n_graphs=extension_count)
    extension = generate_random(extension_config) if extension_count else []
    results = original + extension
    if len({v['scenario_id'] for v in results}) != len(results):
        raise ValueError('Original and extension batches must have distinct scenario IDs')
    for cohort, batch in (('original', original), ('extension', extension)):
        for instance in batch:
            instance['metadata'].update(sample_cohort=cohort, stationary_environment_count=len(results),
                balance_scope='Within each separately sampled batch; combined marginal counts need not differ by at most one')
    return results


def generate_random_switching(instances=None, count=30, selection_seed=2026092401,
                              horizon=30000, L0=50, gamma=1.6):
    """Deterministic common-baseline switches on a declared hash-ranked subset.

    Select the count smallest SHA256(f'{selection_seed}|{stationary_id}')
    values, independently of performance. Low/high means are g-min(g) and
    1-(max(g)-g); M, all gaps and optimal actions remain fixed. Distinct stored
    IDs use '__switching'; metadata rng_scenario_id keeps the original ID so
    a runner can reproduce the saved stationary/switching paired RNG streams.
    Supplied stationary instances receive a matched_switching metadata flag.
    """
    horizon, count = _positive_integer(horizon, 'horizon'), _nonnegative_integer(count, 'count')
    selection_seed = _nonnegative_integer(selection_seed, 'selection_seed')
    if instances is None:
        instances = generate_random_paper(horizon)
    instances = list(instances)
    if count > len(instances):
        raise ValueError('Switching sample cannot exceed the stationary sample')
    by_id = {v['scenario_id']: v for v in instances}
    if len(by_id) != len(instances):
        raise ValueError('Stationary scenario IDs must be unique')
    rank = lambda sid: hashlib.sha256(f'{selection_seed}|{sid}'.encode()).hexdigest()
    selected = sorted(sorted(by_id), key=rank)[:count]
    for instance in instances:
        instance['metadata']['matched_switching'] = instance['scenario_id'] in selected
    starts = geometric_starts(horizon, L0, gamma)
    results = []
    for original_id in selected:
        stationary = by_id[original_id]
        if stationary['metadata']['mode'] != 'stationary' or len(stationary['vectors']) != 1:
            raise ValueError('Switching extension requires stationary source instances')
        M, g0 = stationary['M'], stationary['vectors'][0]
        low, high = g0 - g0.min(), 1. - (g0.max() - g0)
        vectors = np.array([low if k % 2 == 0 else high for k in range(len(starts))])
        metadata = deepcopy(stationary['metadata'])
        metadata.update(mode='deterministic_baseline_switching',
            rng_scenario_id=original_id, switching_source_scenario_id=original_id,
            source_stationary_scenario=original_id, matched_switching=True,
            g0=g0.tolist(), first_segment=int(L0), growth=float(gamma), start_phase='low',
            temporal_change='deterministic_common_baseline_only',
            loss_formula='gL=g0-min(g0); gH=1-(max(g0)-g0)',
            shift_amplitude=float(1 - np.ptp(g0)), clipping_active=False,
            all_gaps_fixed=True, learner_reset_at_switch=False,
            selection_seed=int(selection_seed), selection_rank=rank(original_id),
            selection_rule="Smallest SHA256(f'{selection_seed}|{stationary_id}') values; independent of performance")
        diagnostics = stationary['metadata'].get('complexity_relation') != 'not_computed'
        result = _finish(original_id + '__switching', M, vectors, starts, metadata,
                         horizon, diagnostics=diagnostics)
        result['baseline_groups'] = stationary['baseline_groups'].copy()
        results.append(result)
    return results



def paper_instances(experiment='e1', horizon=30000):
    """Generate the six epsilon=.3 E1 settings or the original 50 random graphs.

    Historical scenario IDs preserve the declared trajectory RNG namespace.
    No datasets are bundled or loaded; all instances are generated from code.
    """
    horizon = _positive_integer(horizon, 'horizon')
    if experiment == 'e1':
        return [instance for A, Z in ((80, 30), (30, 80), (60, 60))
                for instance in generate_e1(A, Z, .3, horizon)]
    if experiment == 'random':
        return generate_random(dict(horizon=horizon))
    raise ValueError("experiment must be 'e1' or 'random'")
