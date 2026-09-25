"""Published stochastic causal-bandit baselines for known post-action mechanisms.

The runner supplies losses, while the papers use rewards. All four methods use
reward = 1 - loss internally. The ``source`` attributes identify the equations.
No method resets at a regime switch. Their stationary guarantees do not extend
to a changing context-loss vector merely because the mechanism is fixed.
"""
from __future__ import annotations

import math
import numpy as np


class IncompatibleMechanismError(ValueError):
    """The named paper algorithm is undefined for the supplied mechanism."""


def _validate_options(options):
    if options:
        raise TypeError("Unknown algorithm options: " + ", ".join(sorted(options)))


def _confidence(value):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence_delta must lie strictly between zero and one")
    return value


class _Baseline:
    def __init__(self, mechanism, horizon, rng):
        self.mechanism = mechanism
        self.M = np.asarray(mechanism.M, dtype=float)
        self.n_actions = int(mechanism.n_actions)
        self.n_contexts = int(mechanism.n_contexts)
        self.horizon = int(horizon)
        self.rng = rng
        if self.horizon < 1 or self.horizon != horizon:
            raise ValueError("horizon must be a positive integer")
        if self.M.shape != (self.n_contexts, self.n_actions):
            raise ValueError("mechanism dimensions are inconsistent")
        if (self.M.size == 0 or not np.isfinite(self.M).all()
                or np.any(self.M < 0)
                or not np.allclose(self.M.sum(axis=0), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("M must contain finite probability columns")

    def _point_mass(self, action):
        p = np.zeros(self.n_actions)
        p[int(action)] = 1.0
        return p

    def _argmax(self, scores):
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite baseline scores")
        best = np.max(scores)
        tied = np.flatnonzero(np.isclose(scores, best, rtol=1e-13, atol=1e-13))
        return int(self.rng.choice(tied))

    def diagnostics(self):
        return {"source": self.source, "stationary_guarantee_only": True}


class CUCB1(_Baseline):
    """Lu et al. (2020), Algorithm 1 C-UCB, fixed-confidence version.

    Reward UCB_z = mean_reward_z + sqrt(2 log(1/delta)/(1 vee N_z)).
    The default delta = T^-2 is the choice in their Theorem 1. Unobserved
    context means are zero; bounds are not clipped (matching Algorithm 1).
    """
    name = "CUCB1"
    source = "https://proceedings.mlr.press/v124/lu20a/lu20a.pdf#page=4"

    def __init__(self, mechanism, horizon, rng, **options):
        super().__init__(mechanism, horizon, rng)
        self.confidence_delta = _confidence(
            options.pop("confidence_delta", 1.0 / max(2, self.horizon) ** 2))
        _validate_options(options)
        self.counts = np.zeros(self.n_contexts, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_contexts)
        self._log_confidence = 2.0 * math.log(1.0 / self.confidence_delta)

    def probabilities(self, t):
        denominator = np.maximum(1, self.counts)
        context_ucb = self.reward_sums / denominator + np.sqrt(
            self._log_confidence / denominator)
        return self._point_mass(self._argmax(self.M.T @ context_ucb))

    def update(self, t, action, context, loss, p):
        self.counts[context] += 1
        self.reward_sums[context] += 1.0 - float(loss)

    def diagnostics(self):
        return {**super().diagnostics(), "confidence_delta": self.confidence_delta,
                "context_counts": self.counts.tolist()}


class CTS(_Baseline):
    """Lu et al. (2020), Algorithm 2 C-TS with independent Beta priors."""
    name = "CTS"
    source = "https://proceedings.mlr.press/v124/lu20a/lu20a.pdf#page=5"

    def __init__(self, mechanism, horizon, rng, **options):
        super().__init__(mechanism, horizon, rng)
        self.prior_alpha = float(options.pop("prior_alpha", 1.0))
        self.prior_beta = float(options.pop("prior_beta", 1.0))
        _validate_options(options)
        if not (np.isfinite(self.prior_alpha) and np.isfinite(self.prior_beta)
                and self.prior_alpha > 0 and self.prior_beta > 0):
            raise ValueError("Beta prior parameters must be positive")
        self.alpha = np.full(self.n_contexts, self.prior_alpha)
        self.beta = np.full(self.n_contexts, self.prior_beta)

    def probabilities(self, t):
        sampled_reward_means = self.rng.beta(self.alpha, self.beta)
        return self._point_mass(self._argmax(self.M.T @ sampled_reward_means))

    def update(self, t, action, context, loss, p):
        reward = 1.0 - float(loss)
        # The paper Bernoulli-resamples bounded real rewards. Here actual
        # rewards are already binary, so this avoids an unnecessary RNG draw.
        success = reward if reward in (0.0, 1.0) else float(self.rng.random() < reward)
        self.alpha[context] += success
        self.beta[context] += 1.0 - success

    def diagnostics(self):
        return {**super().diagnostics(), "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "context_counts": (self.alpha + self.beta - self.prior_alpha
                                   - self.prior_beta).tolist()}


class CUCB2(_Baseline):
    """Nair, Patil and Sinha (2021), Section 5 / Algorithm 3 C-UCB-2.

    c_z = min_a M[z,a]; zeta_a = sum_z M[z,a]/c_z;
    UCB_a(s) = sum_z M[z,a] mean_reward_z(s)
                 + zeta_a sqrt(log(|Z| s^2 / 2)/s).
    Crucially, zeta_a is OUTSIDE the square root. All columns must have
    identical nonzero support. Never replace zero c_z with a numerical floor.
    """
    name = "CUCB2"
    source = "https://arxiv.org/pdf/2012.07058#page=10"

    def __init__(self, mechanism, horizon, rng, **options):
        super().__init__(mechanism, horizon, rng)
        _validate_options(options)
        reachable = np.any(self.M > 0.0, axis=1)
        coverage = self.M.min(axis=1)
        if np.any(coverage[reachable] <= 0.0):
            raise IncompatibleMechanismError(
                "CUCB2 requires all action columns to have identical nonzero "
                "context support (Nair et al., Section 5). This mechanism "
                "violates that assumption. Record this algorithm as "
                "inapplicable; do not fill support zeros with small values.")
        self.coverage = coverage
        ratios = self.M[reachable] / coverage[reachable, None]
        # fsum is invariant to the permutation of a shared-identity column;
        # naive reductions can create spurious bonus differences at tiny eps.
        self.zeta = np.array([math.fsum(ratios[:, a]) for a in range(self.n_actions)])
        if not np.isfinite(self.zeta).all():
            raise FloatingPointError("CUCB2 coverage ratios exceed floating-point range")
        self.counts = np.zeros(self.n_contexts, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_contexts)

    def probabilities(self, t):
        if t <= self.n_actions:
            return self._point_mass(t - 1)
        s = int(t) - 1
        context_means = self.reward_sums / np.maximum(1, self.counts)
        multiplier = math.sqrt(max(0.0, math.log(self.n_contexts * s * s / 2.0)) / s)
        # Subtracting a common bonus preserves argmax and avoids cancellation
        # when all zeta values are large and equal, as in the target mechanism.
        scores = self.M.T @ context_means + multiplier * (self.zeta - self.zeta.min())
        return self._point_mass(self._argmax(scores))

    def update(self, t, action, context, loss, p):
        self.counts[context] += 1
        self.reward_sums[context] += 1.0 - float(loss)

    def diagnostics(self):
        return {**super().diagnostics(), "context_counts": self.counts.tolist(),
                "minimum_context_probabilities": self.coverage.tolist(),
                "zeta": self.zeta.tolist(),
                "common_bonus_across_actions": bool(np.allclose(self.zeta, self.zeta[0]))}


class PECUCB(_Baseline):
    """Liu, Attias and Roy (2024), Algorithm 2 Phased Elimination (PE).

    ``PE-CUCB`` is the experiment label; the source paper calls this ``PE``.
    This is the linear-bandit reduction, not an ad-hoc context-UCB eliminator:
    feature vectors are mechanism columns, and the realized Z_t is ignored.
    Phase regression uses only that phase's rewards.
    """
    name = "PE-CUCB"
    source = "https://arxiv.org/html/2407.00950v1#alg2"

    def __init__(self, mechanism, horizon, rng, **options):
        super().__init__(mechanism, horizon, rng)
        self.confidence_delta = _confidence(
            options.pop("confidence_delta", 1.0 / max(2, self.horizon)))
        self.play_order = options.pop("play_order", "random")
        self.design_tolerance = float(options.pop("design_tolerance", 1e-8))
        self.design_max_iterations = int(options.pop("design_max_iterations", 10000))
        _validate_options(options)
        if self.play_order not in ("random", "block"):
            raise ValueError("PE play_order must be 'random' or 'block'")
        if self.design_tolerance <= 0 or self.design_max_iterations < 1:
            raise ValueError("Design tolerance and maximum iterations must be positive")
        self.dimension = int(np.linalg.matrix_rank(self.M))
        self.active = np.arange(self.n_actions, dtype=int)
        self.phase = 0
        self.phase_history = []
        self._remaining_counts = np.zeros(0, dtype=np.int64)
        self._representatives = np.zeros(0, dtype=int)
        self._remaining_total = 0
        self._scheduled_action = None
        self._scheduled_index = None
        self._position = 0
        self._phase_reward_vector = None
        self._phase_coordinates = None
        self._phase_V = None
        self._phase_meta = {}
        # Source formula is undefined at d=1. All probability columns in a
        # rank-one mechanism are identical, so any action is optimal. The
        # benign extension below chooses uniformly among these equivalent arms.
        self._rank_one = self.dimension == 1
        d = max(2, self.dimension)
        self._phase_base = 4.0 * d * math.log(math.log(d)) + 16.0
        self._confidence_log = math.log(
            2.0 * self.n_actions * max(1.0, math.log2(max(2, self.horizon)))
            / self.confidence_delta)

    @staticmethod
    def _coordinates(features):
        """Whiten the span without changing any prediction or leverage.

        SVD rank uses NumPy's standard numerical-rank threshold. In particular,
        A < Z, duplicate columns, unreachable rows, and rank deficiency are
        allowed. No ridge term is added to the published algorithm.
        """
        u, singular, right = np.linalg.svd(features, full_matrices=False)
        threshold = max(features.shape) * np.finfo(float).eps * singular[0]
        rank = int(np.sum(singular > threshold))
        if rank < 1:
            raise ValueError("No nonzero feature span")
        transform = right[:rank].T / singular[:rank]
        return u[:, :rank], transform, rank

    @staticmethod
    def _leverage(x, weights):
        V = x.T @ (weights[:, None] * x)
        try:
            factor = np.linalg.cholesky(V)
            solved = np.linalg.solve(factor, x.T)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("PE design lost numerical rank") from error
        leverage = np.einsum("ij,ij->j", solved, solved)
        if not np.isfinite(leverage).all():
            raise FloatingPointError("Nonfinite PE design leverage")
        return leverage

    def _compress_design(self, x, weights, limit):
        """Deterministically prune a design and reoptimize its support.

        Each deletion is scored by its worst leverage over ALL active arms.
        The final caller rechecks the 2d certificate, so this numerical
        fallback cannot silently relax the paper's support requirement.
        """
        weights = weights.copy()
        rank = x.shape[1]
        while np.count_nonzero(weights) > limit:
            support = np.flatnonzero(weights)
            V = x.T @ (weights[:, None] * x)
            inverse_x = np.linalg.solve(V, x.T)
            leverage = np.einsum("ij,ji->i", x, inverse_x)
            cross = x @ inverse_x[:, support]
            denom = 1.0 - weights[support] * leverage[support]
            eligible = denom > 1e-10
            if not eligible.any():
                raise RuntimeError("PE cannot remove an arm without losing design rank")
            candidates = support[eligible]
            new_leverage = (1.0 - weights[candidates])[None, :] * (
                leverage[:, None] + cross[:, eligible] ** 2
                * (weights[candidates] / denom[eligible])[None, :])
            removed = int(candidates[np.argmin(new_leverage.max(axis=0))])
            weights[removed] = 0.0
            weights /= weights.sum()
            support = np.flatnonzero(weights)
            # Multiplicative determinant ascent restricted to the retained
            # support. This can alter weights but cannot increase support.
            for _ in range(min(1000, self.design_max_iterations)):
                leverage = self._leverage(x, weights)
                if leverage[support].max() <= rank * (1.0 + self.design_tolerance):
                    break
                weights[support] *= leverage[support] / rank
                weights /= weights.sum()
        return weights

    def _design(self, features):
        """Return a certified sparse 2d G-design for arbitrary active columns.

        Pivoted Gram-Schmidt supplies a full-rank starting design, followed
        by exact line-search determinant ascent. A compression fallback is
        used if necessary. Solver failure is explicit: it is not evidence
        that the mathematical mechanism is incompatible with PE.
        """
        x, _, rank = self._coordinates(features)
        n = len(features)
        support_limit = int(math.floor(self._phase_base))
        target = 2.0 * self.dimension * (1.0 + self.design_tolerance)
        if n == rank:
            weights = np.full(n, 1.0 / n)
            maximum = float(self._leverage(x, weights).max())
            return weights, maximum, rank, 0

        # Select a full-rank initial support by pivoted Gram-Schmidt.
        residual = x.copy()
        support = []
        for _ in range(rank):
            j = int(np.argmax(np.sum(residual * residual, axis=1)))
            support.append(j)
            direction = residual[j] / np.linalg.norm(residual[j])
            residual -= (residual @ direction)[:, None] * direction[None, :]
        weights = np.zeros(n)
        weights[support] = 1.0 / rank
        for iteration in range(self.design_max_iterations + 1):
            leverage = self._leverage(x, weights)
            largest = float(leverage.max())
            if largest <= target:
                if np.count_nonzero(weights) > support_limit:
                    weights = self._compress_design(x, weights, support_limit)
                    leverage = self._leverage(x, weights)
                    largest = float(leverage.max())
                if largest <= target:
                    return weights, largest, rank, iteration
            if iteration == self.design_max_iterations:
                break
            j = int(np.argmax(leverage))
            if rank == 1:
                # Rank-one probability columns are identical. This branch
                # only handles numerical error above the expected certificate.
                raise RuntimeError("PE rank-one design failed its leverage certificate")
            step = (largest - rank) / (rank * (largest - 1.0))
            weights *= 1.0 - step
            weights[j] += step
        raise RuntimeError(
            "PE numerical design solver failed its 2d leverage/sparse-support "
            "certificate. Increase design_max_iterations or inspect conditioning; "
            "no uncertified design was substituted.")

    def _start_phase(self, t):
        self.phase += 1
        unique, inverse = np.unique(self.M[:, self.active].T, axis=0, return_inverse=True)
        # A design needs only one representative of identical feature vectors.
        # Random representatives ensure arm labels are not privileged.
        representatives = np.array([
            self.rng.choice(self.active[inverse == j]) for j in range(len(unique))], dtype=int)
        weights, maximum, rank, iterations = self._design(unique)
        self._phase_size_parameter = (2.0 ** (self.phase - 1)) * self._phase_base
        counts = np.ceil(self._phase_size_parameter * weights).astype(np.int64)
        self._representatives = representatives
        self._remaining_counts = counts.copy()
        self._remaining_total = int(counts.sum())
        self._scheduled_action = None
        self._scheduled_index = None
        self._position = 0
        coordinates, transform, _ = self._coordinates(unique)
        self._phase_coordinates = self.M.T @ transform
        self._phase_V = coordinates.T @ (counts[:, None] * coordinates)
        self._phase_reward_vector = np.zeros(rank)
        self._phase_meta = {"phase": self.phase, "start_round": int(t),
            "planned_rounds": int(counts.sum()), "active_actions_before": len(self.active),
            "active_rank": rank, "design_support": int(np.count_nonzero(weights)),
            "maximum_design_leverage": maximum, "design_iterations": iterations,
            "phase_size_parameter": self._phase_size_parameter}

    def _finish_phase(self, t):
        # Regression in the active span implements Remark 4.4 without
        # numerical pseudoinversion of an ambient singular covariance.
        estimate = np.linalg.solve(self._phase_V, self._phase_reward_vector)
        scores = self._phase_coordinates[self.active] @ estimate
        threshold = 2.0 * math.sqrt(
            4.0 * self.dimension / self._phase_size_parameter * self._confidence_log)
        keep = scores.max() - scores <= threshold + 1e-12
        self.active = self.active[keep]
        self.phase_history.append({**self._phase_meta, "end_round": int(t),
            "elimination_threshold": threshold, "active_actions_after": len(self.active)})

    def probabilities(self, t):
        if self._rank_one:
            return self._point_mass(self.rng.choice(self.active))
        if self._remaining_total == 0:
            self._start_phase(t)
        if self._scheduled_action is None:
            if self.play_order == "random":
                # Sequential sampling without replacement is exactly a
                # uniform random permutation of the phase multiset and
                # needs O(A), rather than O(T), memory.
                draw = int(self.rng.integers(self._remaining_total))
                index = int(np.searchsorted(np.cumsum(self._remaining_counts), draw,
                                            side="right"))
            else:
                index = int(np.flatnonzero(self._remaining_counts)[0])
            self._scheduled_index = index
            self._scheduled_action = int(self._representatives[index])
        return self._point_mass(self._scheduled_action)

    def update(self, t, action, context, loss, p):
        if self._rank_one:
            return
        expected_action = self._scheduled_action
        if int(action) != expected_action:
            raise ValueError("PE received an action different from its scheduled action")
        self._phase_reward_vector += self._phase_coordinates[action] * (1.0 - float(loss))
        self._position += 1
        self._remaining_counts[self._scheduled_index] -= 1
        self._remaining_total -= 1
        self._scheduled_action = None
        self._scheduled_index = None
        if self._remaining_total == 0:
            self._finish_phase(t)

    def diagnostics(self):
        return {**super().diagnostics(), "paper_algorithm_name": "PE",
                "confidence_delta": self.confidence_delta, "play_order": self.play_order,
                "feature_rank": self.dimension, "active_actions": self.active.tolist(),
                "design_relative_tolerance": self.design_tolerance,
                "design_support_limit": int(math.floor(self._phase_base)),
                "rank_one_extension": self._rank_one,
                "completed_phases": self.phase_history,
                "current_phase_rounds_observed": self._position,
                "current_phase": self._phase_meta}
