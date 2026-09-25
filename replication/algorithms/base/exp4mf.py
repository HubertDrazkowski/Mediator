"""EXP4 with mediator feedback, Eldowa et al. (2024).

Algorithm 1, with Theorem 3's BOBW schedule by default. See
https://arxiv.org/html/2402.10282v1#S4.
"""

from __future__ import annotations

import math
import numpy as np


def mechanism_capacity(mechanism):
    """Return (capacity or certified upper bound, exact, description).

    The shared-identity symmetric channel has exact capacity
    (1-epsilon)**2 * (n_contexts-1), including duplicate action columns.
    Other matrices use the certified sum_z max_a M[z,a] - 1 upper bound.
    """
    matrix = np.asarray(mechanism.M, dtype=float)
    n_contexts, n_actions = matrix.shape
    if n_actions == 1 or n_contexts == 1:
        return 0.0, True, "single action or single context"
    if np.array_equal(matrix, np.repeat(matrix[:, :1], n_actions, axis=1)):
        return 0.0, True, "identical mechanism columns"
    dominant = np.argmax(matrix, axis=0)
    epsilon = float(matrix.min()) * n_contexts
    reference = np.full_like(matrix, epsilon / n_contexts)
    reference[dominant, np.arange(n_actions)] += 1.0 - epsilon
    if (
        0 <= epsilon <= 1
        and np.unique(dominant).size == n_contexts
        and np.allclose(matrix, reference, atol=1e-13, rtol=1e-13)
    ):
        return (
            (1.0 - epsilon) ** 2 * (n_contexts - 1),
            True,
            "exact symmetric-channel capacity",
        )
    upper = min(
        float(n_actions - 1),
        float(n_contexts - 1),
        # The equivalent difference form avoids catastrophic cancellation
        # for mechanisms with almost-identical columns.
        max(0.0, math.fsum(matrix.max(axis=1) - matrix.mean(axis=1))),
    )
    return upper, False, "capacity upper bound min(A-1,Z-1,sum_z max_a M[z,a]-1)"


class EXP4MF:
    """Exponential weights with the mediator importance-weighted estimator.

    Parameters
    ----------
    mechanism : object
        Must expose M (contexts by actions), n_actions, n_contexts.
    horizon : int
        T, used by the BOBW schedule; must be fixed before the run.
    rng : numpy.random.Generator
        Accepted for the common runner interface; this learner is deterministic
        conditional on its observations and returns a distribution for sampling.
    schedule : {'bobw', 'adversarial', 'adaptive'}
        Respectively Theorems 3, 1, and 2. No oracle gap or loss means are used.
    capacity : float or None
        Optional known capacity or certified upper bound. By default the exact
        synthetic-channel capacity is inferred, otherwise a bound is used.
    """

    name = "EXP4MF"
    source = "https://arxiv.org/html/2402.10282v1#S4"

    def __init__(self, mechanism, horizon, rng, **options):
        self.mechanism = mechanism
        self.M = np.asarray(mechanism.M, dtype=float)
        self.n_actions = int(mechanism.n_actions)
        self.n_contexts = int(mechanism.n_contexts)
        self.horizon = int(horizon)
        self.rng = rng
        self.schedule = options.pop("schedule", "bobw")
        supplied_capacity = options.pop("capacity", None)
        if options:
            raise TypeError(f"Unknown EXP4MF options: {sorted(options)}")
        if self.horizon < 1 or self.horizon != horizon:
            raise ValueError("horizon must be a positive integer")
        if self.schedule not in {"bobw", "adversarial", "adaptive"}:
            raise ValueError("EXP4MF schedule must be bobw, adversarial, or adaptive")
        if self.M.shape != (self.n_contexts, self.n_actions) or self.M.size == 0:
            raise ValueError("mechanism dimensions are inconsistent")
        if (
            not np.isfinite(self.M).all()
            or (self.M < 0).any()
            or not np.allclose(self.M.sum(axis=0), 1.0, rtol=0, atol=1e-12)
        ):
            raise ValueError("M must have probability distributions as columns")
        cap, exact, description = mechanism_capacity(mechanism)
        if supplied_capacity is not None:
            cap = float(supplied_capacity)
            exact = False
            description = "user-supplied capacity or upper bound"
        if not np.isfinite(cap) or cap < 0:
            raise ValueError("capacity must be finite and nonnegative")
        if cap == 0 and not np.array_equal(self.M, np.repeat(
                self.M[:, :1], self.n_actions, axis=1)):
            raise ValueError("zero capacity requires identical columns")
        self.capacity = cap
        self.capacity_is_exact = exact
        self.capacity_source = description
        self.log_n = math.log(self.n_actions)
        self.gamma = (
            math.sqrt(math.e * cap * (1.0 + math.log(self.horizon)) / (2 * self.log_n))
            if self.n_actions > 1 else 0.0
        )
        self.beta = self.gamma
        self.entropy_sum = 0.0
        self.information_sum = 0.0
        self.cumulative_loss = np.zeros(self.n_actions)
        self.last_eta = 1.0
        self.reachable_contexts = np.any(self.M > 0, axis=1)
        self.underflow_rounds = 0
        self.maximum_underflow_actions = 0
        self.first_underflow_round = None
        self.lost_context_support = []

    def learning_rate(self, t):
        """Rate used before observing round t (t is one-based)."""
        if int(t) != t or t < 1:
            raise ValueError("t must be a positive integer")
        if self.n_actions == 1 or self.capacity == 0:
            return 1.0
        if self.schedule == "bobw":
            return min(1.0, 1.0 / self.beta)
        if self.schedule == "adversarial":
            return min(1.0, math.sqrt(self.log_n / (math.e * self.capacity * t)))
        return math.sqrt(
            self.log_n / (self.log_n + math.e * (self.information_sum + self.capacity))
        )

    def probabilities(self, t):
        self.last_eta = self.learning_rate(t)
        # Shifting every cumulative estimate by its minimum changes no weight.
        logits = -self.last_eta * (self.cumulative_loss - self.cumulative_loss.min())
        if not np.isfinite(logits).all():
            raise FloatingPointError("EXP4MF cumulative-loss logits became nonfinite")
        with np.errstate(under="ignore"):
            weights = np.exp(logits)
        p = weights / weights.sum()
        underflow = int(np.count_nonzero(p == 0))
        if underflow:
            self.underflow_rounds += 1
            self.maximum_underflow_actions = max(self.maximum_underflow_actions, underflow)
            if self.first_underflow_round is None:
                self.first_underflow_round = int(t)
        lost = np.flatnonzero(self.reachable_contexts & ((self.M @ p) == 0))
        if lost.size:
            self.lost_context_support = lost.tolist()
            raise FloatingPointError(
                "EXP4MF floating-point underflow removed reachable context support "
                f"at round {t}: contexts {lost.tolist()}. No probability floor "
                "was inserted; retain this run as a numerical failure.")
        return p

    def diagnostics(self):
        """Return compact JSON-serializable settings and final scalar state."""
        return {
            "source": self.source,
            "schedule": self.schedule,
            "capacity": float(self.capacity),
            "capacity_is_exact": bool(self.capacity_is_exact),
            "capacity_source": self.capacity_source,
            "gamma": float(self.gamma),
            "beta": float(self.beta),
            "last_eta": float(self.last_eta),
            "underflow_rounds": self.underflow_rounds,
            "first_underflow_round": self.first_underflow_round,
            "maximum_underflow_actions": self.maximum_underflow_actions,
            "lost_context_support": self.lost_context_support,
            "probability_floor": None,
        }

    def update(self, t, action, context, loss, p):
        """Use only the observed context/loss and the known mechanism."""
        p = np.asarray(p, dtype=float)
        if p.shape != (self.n_actions,):
            raise ValueError("p has the wrong shape")
        if (not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("p must be the actual action sampling distribution")
        if not np.isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError("loss must lie in [0, 1]")
        context = int(context)
        if not 0 <= context < self.n_contexts:
            raise ValueError("context is outside the mechanism")
        q = self.M @ p
        if q[context] <= 0:
            raise ValueError("an observed context has zero sampling probability")
        # Alg. 1: lhat_t(a)=M[Z_t,a]*loss/q_t(Z_t). No chosen-action IPW.
        # Divide only positive entries: impossible contexts never contribute
        # 0*infinity, and no fictitious denominator floor enters the estimator.
        positive = self.M[context] > 0
        estimate = np.zeros(self.n_actions)
        if loss != 0:
            estimate[positive] = (self.M[context, positive] / q[context]) * float(loss)
        if not np.isfinite(estimate).all():
            raise FloatingPointError("EXP4MF importance estimate exceeded numeric range")
        self.cumulative_loss += estimate
        self.cumulative_loss -= self.cumulative_loss.min()
        if not np.isfinite(self.cumulative_loss).all():
            raise FloatingPointError("EXP4MF cumulative loss exceeded numeric range")
        if self.schedule == "bobw" and self.n_actions > 1:
            positive = p > 0
            entropy = -float(np.dot(p[positive], np.log(p[positive])))
            self.entropy_sum += entropy
            self.beta += self.gamma / math.sqrt(1.0 + self.entropy_sum / self.log_n)
        elif self.schedule == "adaptive" and self.n_actions > 1:
            # Q_p+1 = sum_z,a M[z,a] * Pr(A=a | Z=z). Form the posterior
            # responsibility by multiplying before division; each numerator
            # is <= q_z. The naive chi-square formula can overflow on an
            # action with underflowed p_a and then produce 0*infinity.
            supported = q > 0
            responsibilities = (self.M[supported] * p) / q[supported, None]
            info = math.fsum((self.M[supported] * responsibilities).ravel()) - 1.0
            if not math.isfinite(info):
                raise FloatingPointError("EXP4MF adaptive information exceeded numeric range")
            self.information_sum += max(0.0, info)
