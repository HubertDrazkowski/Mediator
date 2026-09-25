"""Online estimated-mechanism learners with nonretroactive action-loss history.

At round t, p_t, q_t and the loss estimate use the supplied pre-round Mhat_t.
Causal methods accumulate L += Mhat_t.T @ ghat_t, then the runner updates its
counts and calls update_mechanism(Mhat_{t+1}). Historical estimates are never
reprojected through a newer matrix. The current matrix enters the current
regularizer. All learning history and optimizer warm states survive refresh.

EXP4MF refreshes its existing mechanism-capacity calculation and gamma, but
never resets beta, entropy, information or cumulative losses. The new gamma
is used in the next observation's beta increment. These are explicit plug-in
extensions; fixed-known-mechanism theorems are not asserted for estimated M.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import warnings


import numpy as np
from . import registry as previous
from .mechanism import Mechanism
from .base import IncompatibleMechanismError, NumericalSolverError
from .base.tsallis import bonuses, convex_gap_certificate, half_tsallis, loss_estimate
from .base.exp4mf import mechanism_capacity
from .solvers.fast_solver import CandidateFailure, ContextNewton, MinimumNewton
from .solvers.fast_tsallis import _DirectConic

CORE = previous.CORE
ALGORITHM_IDS = (*CORE, "EXP4MF")
BOBW_COMPARISON = ALGORITHM_IDS
COMPARISON_SETS = {"core": CORE, "with_exp4mf": ALGORITHM_IDS}


def algorithm_specs():
    return [deepcopy(spec) for spec in previous.algorithm_specs() if spec["id"] in ALGORITHM_IDS]


specs = algorithm_specs
default_specs = algorithm_specs


def resolve_spec(specification):
    spec = previous.resolve_spec(specification)
    if spec["id"] not in ALGORITHM_IDS:
        raise ValueError("Online study supports only its six declared methods")
    return spec


def applicability(specification, mechanism):
    resolve_spec(specification)
    return None


def _matrix(mechanism, n_contexts, n_actions):
    result = mechanism if isinstance(mechanism, Mechanism) else Mechanism(mechanism)
    if result.M.shape != (n_contexts, n_actions):
        raise ValueError("An online mechanism update cannot change dimensions")
    return result


def _refresh_holder(holder, mechanism):
    """Refresh geometry only; no cumulative estimates or warm states reset."""
    holder.mechanism, holder.M = mechanism, mechanism.M
    reachable = holder.M[np.any(holder.M > 0, axis=1)]
    entropy = np.sqrt(holder.M).sum(axis=0)
    holder._reachable_M, holder._fast_h = reachable, entropy
    # Past action costs need not agree across currently identical columns, so
    # the fixed-M dominant-column reduction must not merge these coordinates.
    holder._dominant = None
    holder._problem = None
    holder._fast_conic = None
    if getattr(holder, "_context_newton", None) is not None:
        holder._context_newton.M = reachable
    if getattr(holder, "_fast_newton", None) is not None:
        holder._fast_newton.M, holder._fast_newton.h = reachable, entropy
    dual = getattr(holder, "_noalpha", None)
    if dual is not None:
        dual.M, dual.reachable, dual.h = holder.M, reachable, entropy
        dual.context.M = reachable
        dual.smooth.M, dual.smooth.h = reachable, entropy
        dual.context_learner.M = holder.M
        dual.context_learner.mechanism = mechanism
        dual.context_learner._problem = None
        dual.context_conic = None


class _OnlineBase:
    def __init__(self, spec, mechanism, horizon, rng):
        self.spec = deepcopy(spec)
        constructed = previous.instantiate(spec, mechanism, horizon, rng)
        # Context's fixed-M failure proxy calls its old G-based objective.
        # This adapter instead supplies action costs to the same solvers and
        # independently certifies every fallback, so unwrap that proxy.
        self.learner = getattr(constructed, "learner", constructed)
        self._last_loss_round = 0
        self._last_refresh_round = 0
        self._model_updates = 0

    def __getattr__(self, name):
        return getattr(self.learner, name)

    @property
    def cumulative_action_estimates(self):
        if "L" in self.__dict__:
            return self.L
        if hasattr(self.learner, "cumulative_loss"):
            return self.learner.cumulative_loss
        return self.learner.L

    def _check_decision(self, t):
        if int(t) != t or t != self._last_loss_round + 1:
            raise ValueError("Online rounds must be sequential")
        if self._last_refresh_round != self._last_loss_round:
            raise ValueError("Refresh the mechanism after the preceding observation before the next decision")

    def _check_refresh(self, mechanism):
        if self._last_loss_round != self._last_refresh_round + 1:
            raise ValueError("update_mechanism must follow exactly one completed loss update")
        return _matrix(mechanism, self.n_contexts, self.n_actions)

    def _finish_refresh(self):
        self._last_refresh_round = self._last_loss_round
        self._model_updates += 1

    def _online_diagnostics(self):
        return dict(online_mechanism_updates=self._model_updates,
                    last_loss_round=self._last_loss_round, last_mechanism_refresh_round=self._last_refresh_round,
                    loss_history_space="action", historical_estimates_reprojected=False,
                    update_order="old-Mhat loss update; observe pair; refresh Mhat for next round",
                    fixed_known_mechanism_guarantee_asserted=False)


class OnlineINF(_OnlineBase):
    def probabilities(self, t):
        self._check_decision(t)
        return self.learner.probabilities(t)

    def update(self, t, action, context, loss, p):
        self._check_decision(t)
        self.learner.update(t, action, context, loss, p)
        self._last_loss_round = int(t)

    def update_mechanism(self, mechanism):
        model = self._check_refresh(mechanism)
        self.learner.mechanism, self.learner.M = model, model.M
        self._finish_refresh()

    def diagnostics(self):
        return {**self.learner.diagnostics(), **self._online_diagnostics(), "learner_uses_M": False}


class OnlineEXP4MF(_OnlineBase):
    def __init__(self, *args):
        super().__init__(*args)
        self.initial_capacity = float(self.learner.capacity)
        self.initial_gamma = float(self.learner.gamma)

    def probabilities(self, t):
        self._check_decision(t)
        return self.learner.probabilities(t)

    def update(self, t, action, context, loss, p):
        self._check_decision(t)
        self.learner.update(t, action, context, loss, p)
        self._last_loss_round = int(t)

    def update_mechanism(self, mechanism):
        model = self._check_refresh(mechanism)
        capacity, exact, description = mechanism_capacity(model)
        learner = self.learner
        learner.mechanism, learner.M = model, model.M
        learner.reachable_contexts = np.any(model.M > 0, axis=1)
        learner.capacity, learner.capacity_is_exact, learner.capacity_source = capacity, exact, description
        learner.gamma = (math.sqrt(math.e * capacity * (1. + math.log(learner.horizon)) / (2 * learner.log_n))
                         if learner.n_actions > 1 else 0.)
        # beta deliberately remains untouched; gamma enters the next update.
        self._finish_refresh()

    def diagnostics(self):
        return {**self.learner.diagnostics(), **self._online_diagnostics(),
                "initial_capacity": self.initial_capacity, "initial_gamma": self.initial_gamma,
                "capacity_refresh_rule": "existing mechanism_capacity on current Mhat; beta history preserved"}


class OnlineCausal(_OnlineBase):
    """Action-history FTRL with current-Mhat geometry and certified solvers."""
    def __init__(self, *args):
        super().__init__(*args)
        self.L = np.zeros(self.n_actions)
        self.last_action_loss_estimate = np.zeros(self.n_actions)
        self._online_solver_counts = CounterLike()
        self._online_solver_failures = CounterLike()
        self._maximum_gap = 0.
        self._online_conic = None
        self._is_action = self.spec["id"] == "CTsallis-Action"
        if not self._is_action:
            _refresh_holder(self.learner, self.learner.mechanism)
            self.learner._context_newton.warm = np.full(self.n_actions, 1 / self.n_actions)
            self.learner._context_newton.max_iterations = 100

    @property
    def cumulative_action_loss(self):
        return self.L

    def update(self, t, action, context, loss, p):
        self._check_decision(t)
        p = np.asarray(p, dtype=float)
        if p.shape != (self.n_actions,) or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum() - 1) > 1e-12:
            raise ValueError("Invalid sampling distribution")
        q = self.M @ p
        if np.any((q <= 0) & np.any(self.M > 0, axis=1)):
            raise NumericalSolverError("Online causal update lost reachable context support")
        ghat = loss_estimate(context, loss, q, self.eta_scale / math.sqrt(t), self.estimator, self.learner._groups)
        increment = self.M.T @ ghat
        if not np.isfinite(increment).all():
            raise NumericalSolverError("Online causal action estimate overflowed")
        self.last_action_loss_estimate = increment.copy()
        self.L += increment
        self.L -= self.L.min()
        if not np.isfinite(self.L).all():
            raise NumericalSolverError("Online cumulative action losses overflowed")
        self._last_loss_round = int(t)

    def update_mechanism(self, mechanism):
        model = self._check_refresh(mechanism)
        if self._is_action:
            self.learner.mechanism, self.learner.M = model, model.M
        else:
            _refresh_holder(self.learner, model)
            self._online_conic = None
        self._finish_refresh()

    def _checked_candidate(self, p, costs, weight, auxiliary_gap=None):
        holder = self.learner
        p = np.asarray(p, dtype=float).copy()
        if not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum() - 1) > 1e-12:
            raise CandidateFailure("Online solver returned an invalid simplex point")
        p[np.argmax(p)] += 1 - p.sum()
        if np.any(p < 0) or np.any(holder._reachable_M @ p <= 0):
            raise CandidateFailure("Online solver lost required context support")
        if holder.geometry == "minimum" and holder.lambda_alpha > 0 and np.any(p <= 0):
            raise CandidateFailure("Online alpha regularization lost action support")
        gap = holder._certificate(p, costs, weight)
        if auxiliary_gap is not None:
            gap = min(gap, float(auxiliary_gap))
        if not np.isfinite(gap) or gap > holder.fast_gap_tolerance:
            raise CandidateFailure(f"Online objective gap {gap} exceeds {holder.fast_gap_tolerance}")
        return p, float(gap)

    def _conic_fallback(self, costs):
        holder = self.learner
        failures = []
        try:
            if self._online_conic is None:
                self._online_conic = _DirectConic(holder)
            p, iterations, status = self._online_conic.solve(costs)
            certificate = convex_gap_certificate(holder.M, p, costs, holder.geometry, holder.context_bonus_scale,
                getattr(holder, "alpha", 2/3), getattr(holder, "lambda_alpha", 0.))
            weight = certificate.get("certificate_action_weight", 0.)
            bound = min(certificate["normalized_global_gap_bound"], self._online_conic.certificate(holder, p, costs))
            p, gap = self._checked_candidate(p, costs, weight, bound)
            return p, gap, weight, iterations, "direct_conic"
        except (CandidateFailure, NumericalSolverError, FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(str(error))
            self._online_solver_failures.increment("direct_conic")
        # Use existing verified primal/dual and polishing helpers with the
        # supplied action costs. Calling the legacy probabilities method would
        # incorrectly replace these costs by Mhat_current.T @ historical G.
        import cvxpy as cp
        for polish in (False, True):
            try:
                holder._build_problem()
                holder._cost.value = costs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    holder._problem.solve(solver="CLARABEL", warm_start=False,
                        tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10,
                        max_iter=holder.solver_max_iterations, static_regularization_constant=1e-12)
                if holder._p.value is None:
                    raise CandidateFailure("Cold online conic solve returned no policy")
                p, _, detail = holder._verified_candidate(1e-7)
                if polish:
                    p, _ = holder._trust_polish_candidate(p)
                certificate = convex_gap_certificate(holder.M, p, costs, holder.geometry,
                    holder.context_bonus_scale, getattr(holder, "alpha", 2/3), getattr(holder, "lambda_alpha", 0.))
                weight = certificate.get("certificate_action_weight", detail.get("action_branch_weight", 0.))
                bound = min(certificate["normalized_global_gap_bound"], holder._conic_dual_certificate(p)["normalized_global_gap_bound"])
                p, gap = self._checked_candidate(p, costs, weight, bound)
                return p, gap, weight, int(holder._problem.solver_stats.num_iters), "cold_conic_polish" if polish else "cold_conic"
            except (cp.error.SolverError, CandidateFailure, NumericalSolverError, FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                failures.append(str(error))
        raise NumericalSolverError("Online current-geometry solve failed without changing the objective/tolerance: " + "; ".join(failures))

    def probabilities(self, t):
        self._check_decision(t)
        holder = self.learner
        if self._is_action:
            p, holder._dual = half_tsallis(self.L / (holder.c * math.sqrt(t)), holder._dual)
            holder._last = dict(solver="scalar KKT on cumulative action costs", main_coefficient_c=holder.c,
                                half_tsallis_dual=holder._dual)
            return p
        costs = (self.L - self.L.min()) / holder._coefficient(t)
        weight, iterations = 0., 0
        try:
            if holder.geometry == "context":
                try:
                    p, iterations = holder._context_newton.solve(costs)
                except (CandidateFailure, FloatingPointError, ValueError, np.linalg.LinAlgError):
                    refinement = ContextNewton(holder.M, beta=holder.context_bonus_scale,
                        tolerance=min(1e-8, holder.fast_gap_tolerance/20), max_iterations=500)
                    refinement.warm = holder._context_newton.warm.copy()
                    p, iterations = refinement.solve(costs)
                method = "context_newton"
                auxiliary = None
            elif holder.lambda_alpha == 0:
                p, certificate = holder._noalpha.solve(costs)
                weight = certificate["certificate_action_weight"]
                auxiliary = certificate["normalized_global_gap_bound"]
                iterations = certificate.get("noalpha_dual_iterations", 0)
                method = "noalpha_dual"
            else:
                if holder._fast_newton is None:
                    holder._fast_newton = MinimumNewton(holder.M, lam=holder.lambda_alpha, alpha=holder.alpha,
                        context_scale=holder.context_bonus_scale, tolerance=holder.fast_gap_tolerance/2)
                before = holder._fast_newton.total_iterations
                p, certificate = holder._fast_newton.solve(costs)
                iterations = holder._fast_newton.total_iterations - before
                weight = certificate["certificate_action_weight"]
                auxiliary = certificate["normalized_global_gap_bound"]
                method = "minimum_newton"
            p, gap = self._checked_candidate(p, costs, weight, auxiliary)
        except (CandidateFailure, NumericalSolverError, FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            self._online_solver_failures.increment("newton")
            p, gap, weight, iterations, method = self._conic_fallback(costs)
        self._online_solver_counts.increment(method)
        self._maximum_gap = max(self._maximum_gap, gap)
        if holder.geometry == "context":
            holder._context_newton.warm = p.copy()
        elif holder.lambda_alpha == 0:
            holder._noalpha.warm, holder._noalpha.weight = p.copy(), weight
            holder._noalpha.context.warm = p.copy()
        elif holder._fast_newton is not None:
            holder._fast_newton.warm, holder._fast_newton.weight = p.copy(), weight
        holder._last = dict(solver="online_certified_" + method, solver_status="certified",
            normalized_global_gap_bound=gap, maximum_certified_normalized_gap=self._maximum_gap,
            unnormalized_global_gap_bound=gap * holder._coefficient(t), solver_iterations=int(iterations),
            fast_solver_counts=dict(self._online_solver_counts), fast_solver_fallbacks=dict(self._online_solver_failures),
            minimum_action_probability=float(p.min()), minimum_reachable_context_probability=float((holder._reachable_M @ p).min()),
            context_bonus_scale=holder.context_bonus_scale, **bonuses(holder.M, p, getattr(holder, "alpha", 2/3)))
        if holder.geometry == "minimum":
            holder._last.update(action_branch_weight=weight, context_branch_weight=1-weight, jensen_action_weight=weight)
        return p

    def diagnostics(self):
        return {**self.learner.diagnostics(), **self._online_diagnostics(),
                "cumulative_action_loss_sha256": hashlib.sha256(self.L.tobytes()).hexdigest(),
                "maximum_certified_normalized_gap": self._maximum_gap if not self._is_action else None}


class CounterLike(dict):
    def increment(self, key):
        self[key] = self.get(key, 0) + 1


def instantiate(specification, mechanism, horizon, rng):
    spec = resolve_spec(specification)
    cls = OnlineINF if spec["id"] == "Tsallis-INF" else OnlineEXP4MF if spec["id"] == "EXP4MF" else OnlineCausal
    return cls(spec, mechanism, horizon, rng)


def resolved_parameters(specification, learner):
    spec = resolve_spec(specification)
    result = previous.resolved_parameters(spec, learner.learner)
    result.update(mechanism_protocol="pre-round online estimate; refresh after observed loss update",
        cumulative_loss_protocol="sum of per-round action estimates, never retroactively reprojected",
        regularizer_mechanism="current supplied Mhat", learner_uses_M=spec["id"] != "Tsallis-INF",
        learning_state_reset_on_refresh=False, fixed_known_mechanism_guarantee_asserted=False)
    if spec["id"] == "EXP4MF":
        result.update(capacity_refresh_rule="recompute existing mechanism_capacity and gamma each round; preserve beta, entropy, information and losses",
                      initial_capacity=learner.initial_capacity, initial_gamma=learner.initial_gamma)
    return result
