"""Prequential count-based mechanism estimates; no true-M learner input.

The matrix issued for round t uses the offline sample and online observations
strictly before t. Call observe(t, action, context) only AFTER the learner's
round-t loss update. Observations are adaptive: no extra uniform online action
sampling is introduced. Every action column retains total prior strength one.

Updating M does not restore unbiased importance weighting at finite time, and
adaptively neglected actions need not obtain accurate columns. These are
plug-in robustness experiments rather than a new known-M regret guarantee.
"""
from __future__ import annotations

from numbers import Integral, Real
import hashlib
import json

import numpy as np


def calibration_identity(M, seed, epsilon=.3):
    """Original independent calibration namespace; seed identifies a replicate."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or min(M.shape) < 1:
        raise ValueError('M must be a nonempty Z-by-A matrix')
    seed = _integer(seed, 'seed')
    identity = dict(namespace='ablation_mhat_calibration_v1', A=M.shape[1],
                    Z=M.shape[0], epsilon=float(epsilon), seed=seed)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                      separators=(',', ':')).encode()).digest()
    return dict(identity, calibration_id='cal_' + digest.hex()[:24],
                calibration_seed=int.from_bytes(digest[:16], 'little'))


def sample_calibration(M, seed, n_max=2000, epsilon=.3):
    """Return iid uniform actions and conditional contexts from true M.

    Draw n_max=2000 once and take the first 500 pairs for the smaller estimate.
    This function is simulator-side: only its observed pairs are given to the
    estimator. Separate RNG streams preserve the original paired benchmark.
    """
    M = np.asarray(M, dtype=float)
    n_max = _integer(n_max, 'n_max', minimum=1)
    if (M.ndim != 2 or not np.isfinite(M).all() or np.any(M < 0)
            or not np.allclose(M.sum(axis=0), 1., atol=1e-12, rtol=0)):
        raise ValueError('M must be finite, nonnegative and column stochastic')
    identity = calibration_identity(M, seed, epsilon)
    action_seed, context_seed = np.random.SeedSequence(identity['calibration_seed']).spawn(2)
    actions = np.random.default_rng(action_seed).integers(M.shape[1], size=n_max, dtype=np.int64)
    uniforms = np.random.default_rng(context_seed).random(n_max)
    cumulative = np.cumsum(M, axis=0)
    cumulative /= cumulative[-1]
    contexts = np.empty(n_max, dtype=np.int64)
    for action in np.unique(actions):
        selected = actions == action
        contexts[selected] = np.searchsorted(cumulative[:, action], uniforms[selected], side='right')
    return actions, contexts


def count_pairs(actions, contexts, n_actions, n_contexts, n=None):
    """Truth-free Z-by-A integer counts, optionally for a nested prefix."""
    A = _integer(n_actions, 'n_actions', minimum=1)
    Z = _integer(n_contexts, 'n_contexts', minimum=1)
    a, z = np.asarray(actions), np.asarray(contexts)
    if a.ndim != 1 or a.shape != z.shape or not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(z.dtype, np.integer):
        raise ValueError('Action/context arrays must have equal one-dimensional integer shapes')
    if n is not None:
        n = _integer(n, 'n')
        if n > len(a):
            raise ValueError('Prefix exceeds available observed pairs')
        a, z = a[:n], z[:n]
    if np.any((a < 0) | (a >= A)) or np.any((z < 0) | (z >= Z)):
        raise ValueError('Observed labels are outside the declared dimensions')
    return np.bincount(z.astype(np.int64) * A + a, minlength=Z * A).reshape(Z, A)


def estimate_from_counts(counts, prior_strength=1.):
    """Symmetric smoothing: (N[z,a]+prior/Z)/(N[a]+prior)."""
    return OnlineMechanismEstimate(counts, prior_strength).matrix(1).copy()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly_copy(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def estimation_error(estimated_matrix, true_matrix):
    """Evaluator-only errors. Neither true matrix nor errors are retained."""
    estimated = np.asarray(estimated_matrix, dtype=float)
    true = np.asarray(true_matrix, dtype=float)
    if estimated.ndim != 2 or estimated.shape != true.shape or min(estimated.shape) < 1:
        raise ValueError("Both matrices must have the same nonempty Z-by-A shape")
    for matrix in (estimated, true):
        if not np.isfinite(matrix).all() or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=0), 1., atol=1e-12, rtol=0):
            raise ValueError("Diagnostics require finite column-stochastic matrices")
    difference = estimated - true
    errors = np.abs(difference).sum(axis=0)
    return dict(mean_column_l1=float(errors.mean()), max_column_l1=float(errors.max()),
                frobenius_error=float(np.linalg.norm(difference)),
                max_absolute_error=float(np.abs(difference).max()))


class OnlineMechanismEstimate:
    """An offline-warm-started estimator updated by completed online pairs.

    matrix(t) returns the immutable matrix for the next round t. Previously
    issued matrices remain unchanged after observe. Updating one action column
    costs one matrix copy plus O(Z) arithmetic; unchanged columns are not refit.
    Counts and diagnostic snapshots are independent read-only copies.
    """

    def __init__(self, initial_counts, prior_strength=1.):
        counts = np.asarray(initial_counts)
        if counts.ndim != 2 or min(counts.shape) < 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("offline_counts must be a nonempty integer Z-by-A matrix")
        if np.any(counts < 0) or np.any(counts > np.iinfo(np.int64).max):
            raise ValueError("offline_counts must contain nonnegative int64 counts")
        if isinstance(prior_strength, (bool, np.bool_)) or not isinstance(prior_strength, Real):
            raise ValueError("prior_strength must be finite and positive")
        prior_strength = float(prior_strength)
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be finite and positive")
        self._counts = np.array(counts, dtype=np.int64, copy=True)
        self._initial_counts = _readonly_copy(self._counts)
        self.n_contexts, self.n_actions = self._counts.shape
        self.prior_strength = prior_strength
        self._pseudocount = prior_strength / self.n_contexts
        if self._pseudocount == 0:
            raise ValueError("prior_strength is too small to represent positive probabilities")
        self._action_counts = self._counts.sum(axis=0)
        if np.any(self._action_counts < 0):
            raise ValueError("offline action counts overflow int64")
        self.offline_samples = int(self._counts.sum())
        self.online_observations = 0
        self._matrix = (self._counts + self._pseudocount) / (self._action_counts + self.prior_strength)
        if np.any(self._matrix <= 0) or not np.isfinite(self._matrix).all():
            raise ValueError("prior_strength is too small to preserve positive finite probabilities")
        self._matrix.setflags(write=False)
        self._last_issued_round = None

    @property
    def counts(self):
        return _readonly_copy(self._counts)

    @property
    def initial_counts(self):
        return _readonly_copy(self._initial_counts)

    @property
    def online_counts(self):
        return _readonly_copy(self._counts - self._initial_counts)

    @property
    def action_counts(self):
        return _readonly_copy(self._action_counts)

    def matrix(self, t):
        """Return Mhat_t using exactly t-1 completed online observations."""
        t = _integer(t, "t", minimum=1)
        if t != self.online_observations + 1:
            raise ValueError("matrix(t) requires exactly t-1 completed online observations")
        self._last_issued_round = t
        # A view prevents consumers from setting this immutable owned snapshot
        # writeable. observe replaces the owned snapshot rather than mutating it.
        view = self._matrix.view()
        view.setflags(write=False)
        return view

    def observe(self, t, action, context):
        """Append a round's pair after its learner loss update has completed."""
        t = _integer(t, "t", minimum=1)
        action = _integer(action, "action")
        context = _integer(context, "context")
        if t != self.online_observations + 1 or self._last_issued_round != t:
            raise ValueError("observe requires the issued next-round matrix and one sequential update")
        if action >= self.n_actions or context >= self.n_contexts:
            raise ValueError("Observed action/context is outside the declared labels")
        if self._action_counts[action] >= np.iinfo(np.int64).max:
            raise OverflowError("Online count exceeds int64")
        self._counts[context, action] += 1
        self._action_counts[action] += 1
        following = self._matrix.copy()
        following[:, action] = ((self._counts[:, action] + self._pseudocount) /
                                (self._action_counts[action] + self.prior_strength))
        if np.any(following[:, action] <= 0) or not np.isfinite(following[:, action]).all():
            raise FloatingPointError("Updated posterior lost positive finite probabilities")
        following.setflags(write=False)
        self._matrix = following
        self.online_observations += 1
        self._last_issued_round = None

    def snapshot(self, t):
        """Independent pre-round state for persistence, without evaluator truth."""
        return dict(t=_integer(t, "t", minimum=1),
                    online_observations=self.online_observations,
                    total_observations=self.offline_samples + self.online_observations,
                    M_hat=_readonly_copy(self.matrix(t)), counts=self.counts,
                    action_counts=self.action_counts)

    def diagnostics(self, true_matrix):
        """Evaluator-only accuracy of the next-round matrix, never stored."""
        return dict(offline_samples=self.offline_samples,
                    online_observations=self.online_observations,
                    total_observations=self.offline_samples + self.online_observations,
                    prior_strength=self.prior_strength,
                    unseen_actions=int(np.count_nonzero(self._action_counts == 0)),
                    minimum_action_count=int(self._action_counts.min()),
                    maximum_action_count=int(self._action_counts.max()),
                    **estimation_error(self._matrix, true_matrix))
