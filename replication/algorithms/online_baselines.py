"""Online plug-in extensions for the missing full-method comparison cells.

CUCB/CTS use the current fitted matrix; CUCB2 also refreshes its coverage bonus.
PE fits M after every observation but freezes its regression features and
sampling design within each phase, adopting the current fit at phase starts.
No historical rewards, active sets, posteriors, or learning rates are reset.
"""
from copy import deepcopy
import math

import numpy as np
from . import registry
from .online import OnlineEXP4MF, _matrix
from .base.exp4mf import EXP4MF

IDS = ('EXP4MF', 'EXP4MF-RV', 'PECUCB', 'CUCB', 'CTS', 'CUCB2')


class _ZeroSafeRate:
    def learning_rate(self, t):
        # The uniform zero-data model has capacity=gamma=beta=0.
        # min(1,1/beta) has the continuous value 1 at beta=0.
        if self.schedule == 'bobw' and self.beta == 0 and t >= 1:
            return 1.0
        return super().learning_rate(t)


class _IW(_ZeroSafeRate, EXP4MF):
    pass


class _RW(_ZeroSafeRate, registry.EXP4MFRV):
    pass


def specs():
    return [s for s in registry.algorithm_specs() if s['id'] in IDS]


class OnlineEXP4MFRW(OnlineEXP4MF):
    def __init__(self, spec, mechanism, horizon, rng):
        self.spec = deepcopy(spec)
        cls = _RW if spec['id'] == 'EXP4MF-RV' else _IW
        self.learner = cls(mechanism, horizon, rng, **deepcopy(spec['options']))
        self._last_loss_round = self._last_refresh_round = self._model_updates = 0
        self.initial_capacity = float(self.learner.capacity)
        self.initial_gamma = float(self.learner.gamma)


class OnlineBaseline:
    def __init__(self, spec, mechanism, horizon, rng):
        self.spec = deepcopy(spec)
        self.learner = registry.instantiate(spec, mechanism, horizon, rng)
        self._last_loss_round = self._last_refresh_round = 0
        self.policy_mechanism_t = 1
        self.model_adoptions = 1

    def __getattr__(self, name):
        return getattr(self.learner, name)

    @property
    def cumulative_action_estimates(self):
        # These learners do not maintain importance-weighted action losses.
        return np.empty(0, dtype=float)

    def probabilities(self, t):
        if t != self._last_loss_round + 1 or self._last_refresh_round != t-1:
            raise ValueError('Sequential online decision/update order violated')
        return self.learner.probabilities(t)

    def update(self, t, action, context, loss, p):
        if t != self._last_loss_round + 1:
            raise ValueError('Nonsequential observation')
        self.learner.update(t, action, context, loss, p)
        self._last_loss_round = int(t)

    def update_mechanism(self, mechanism):
        if self._last_loss_round != self._last_refresh_round + 1:
            raise ValueError('Model refresh must follow one observation')
        model = _matrix(mechanism, self.n_contexts, self.n_actions)
        inner = self.learner
        method = self.spec['id']
        adopt = method != 'PECUCB' or inner._remaining_total == 0
        if adopt:
            changed = not np.array_equal(inner.M, model.M)
            inner.mechanism, inner.M = model, model.M
            self.policy_mechanism_t = self._last_loss_round + 1
            self.model_adoptions += 1
            if method == 'CUCB2' and changed:
                reachable = np.any(model.M > 0, axis=1)
                inner.coverage = model.M.min(axis=1)
                if np.any(inner.coverage[reachable] <= 0):
                    raise ValueError('CUCB2 estimated support is incompatible')
                ratios = model.M[reachable] / inner.coverage[reachable, None]
                inner.zeta = np.array([math.fsum(ratios[:, a]) for a in range(self.n_actions)])
            if method == 'PECUCB' and changed:
                # Keep the completed phase's geometry through its elimination.
                # The next design uses only the new pre-round model.
                inner.dimension = int(np.linalg.matrix_rank(model.M))
                inner._rank_one = inner.dimension == 1
                d = max(2, inner.dimension)
                inner._phase_base = 4.0*d*math.log(math.log(d)) + 16.0
        self._last_refresh_round = self._last_loss_round

    def diagnostics(self):
        return {**self.learner.diagnostics(),
            'online_mechanism_updates': self._last_refresh_round,
            'policy_mechanism_t': self.policy_mechanism_t,
            'model_adoptions': self.model_adoptions,
            'mechanism_refresh_rule': 'phase boundaries' if self.spec['id']=='PECUCB' else 'every round',
            'fixed_known_mechanism_guarantee_asserted': False,
            'has_importance_weighted_action_loss_history': False}


def instantiate(specification, mechanism, horizon, rng):
    if specification['id'] not in IDS:
        raise ValueError('Unsupported new-run method')
    if specification['id'] in ('EXP4MF', 'EXP4MF-RV'):
        return OnlineEXP4MFRW(specification, mechanism, horizon, rng)
    return OnlineBaseline(specification, mechanism, horizon, rng)


def resolved_parameters(specification, learner):
    params = registry.resolved_parameters(specification, learner.learner)
    params.update(online_estimation=True, learning_state_reset_on_refresh=False,
        fixed_known_mechanism_guarantee_asserted=False,
        mechanism_refresh_rule='at each phase start' if specification['id']=='PECUCB' else 'every round')
    return params
