"""Tsallis learners on arbitrary, known action--context mechanisms.

No learner is passed losses, gaps, an optimal action, or a switching schedule.
All rounds are one-based. General context objectives are solved over the full
action simplex, so unattainable context distributions are never optimized.
"""
from __future__ import annotations

import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np


class NumericalSolverError(RuntimeError):
    """The requested objective could not be solved to its declared tolerance."""


def half_tsallis(costs: np.ndarray, warm: float | None = None) -> tuple[np.ndarray, float]:
    """Solve min costs@p - sum(sqrt(p)), exactly up to a scalar KKT tolerance."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not len(costs) or not np.isfinite(costs).all():
        raise ValueError("costs must be a finite, nonempty vector")
    if len(costs) == 1:
        return np.ones(1), 0.5
    c = costs - costs.min()
    lower, upper = 0.0, 0.5 * math.sqrt(len(c))
    dual = warm if warm is not None and 0 < warm < upper else upper / 2
    for _ in range(160):
        denominator = c + dual
        p = (0.5 / denominator) ** 2
        residual = float(p.sum() - 1)
        if abs(residual) <= 2e-13:
            p[np.argmax(p)] -= residual
            if np.any(p <= 0):
                raise NumericalSolverError("Positive half-Tsallis support underflowed")
            return p, float(dual)
        if residual > 0:
            lower = dual
        else:
            upper = dual
        slope = -2 * float(np.sum(p / denominator))
        proposed = dual - residual / slope
        dual = proposed if lower < proposed < upper else (lower + upper) / 2
    raise NumericalSolverError("Half-Tsallis KKT root failed to converge")


def rv_baseline(probabilities: np.ndarray, eta: float,
                groups: np.ndarray | None = None) -> np.ndarray:
    """Predictable baseline; coarse groups affect only the threshold decision."""
    probabilities = np.asarray(probabilities, dtype=float)
    threshold_probabilities = probabilities
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != probabilities.shape:
            raise ValueError("baseline_groups must have one label per context")
        _, inverse = np.unique(groups, return_inverse=True)
        threshold_probabilities = np.bincount(inverse, weights=probabilities)[inverse]
    return 0.5 * (threshold_probabilities >= eta * eta)


def loss_estimate(index: int, loss: float, probabilities: np.ndarray, eta: float,
                  estimator: str = "rv", groups: np.ndarray | None = None) -> np.ndarray:
    """RV/IW with the actual sampling probability, never an artificial floor.

    Conditional unbiasedness holds for coordinates with positive probability.
    A zero-probability coordinate cannot be estimated from the current draw.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if not 0 <= int(index) < len(probabilities) or not np.isfinite(loss) or not 0 <= loss <= 1:
        raise ValueError("Invalid observed index/loss")
    if probabilities[index] <= 0:
        raise ValueError("Observed outcome has zero sampling probability")
    if estimator == "rv":
        estimate = rv_baseline(probabilities, eta, groups)
    elif estimator == "iw":
        estimate = np.zeros_like(probabilities)
    else:
        raise ValueError("estimator must be 'rv' or 'iw'")
    estimate[index] += (float(loss) - estimate[index]) / probabilities[index]
    if not np.isfinite(estimate).all():
        raise NumericalSolverError("Importance estimate overflowed; no floor was substituted")
    return estimate


def bonuses(matrix: np.ndarray, p: np.ndarray, alpha: float = 2 / 3) -> dict[str, float]:
    """Unscaled bonuses used in the objective and logged branch diagnostics."""
    matrix, p = np.asarray(matrix), np.asarray(p)
    h = np.sqrt(matrix).sum(axis=0)
    return {
        "action_bonus": float(np.sqrt(p).sum() - 1),
        "raw_context_bonus": float(np.sqrt(matrix @ p).sum()),
        "jensen_bonus": float(np.sqrt(matrix @ p).sum() - h @ p),
        "alpha_bonus": float((np.power(p, alpha).sum() - 1) / (alpha * (1 - alpha))),
    }


def convex_gap_certificate(matrix, p, costs, geometry, context_scale=1.0,
                           alpha=2 / 3, lambda_alpha=0.0, branch_weight=0.5):
    """Independent global objective-gap upper bound on the action simplex.

    For the minimum regularizer, every w in [0,1] supplies a convex lower
    objective f_w <= f. Its tangent gives min(f) >= f_w(p)-g.p+min(g).
    The returned f(p)-lower_bound is therefore a global certificate, even
    when p is at a branch intersection or an action face (w=0).
    """
    matrix, p, costs = np.asarray(matrix), np.asarray(p), np.asarray(costs)
    q = matrix @ p
    reachable = np.any(matrix > 0, axis=1)
    if np.any(q[reachable] <= 0) or np.any(p < 0):
        return {"normalized_global_gap_bound": math.inf}
    gradient_context = matrix[reachable].T @ (0.5 / np.sqrt(q[reachable]))
    gradient_jensen = context_scale * (gradient_context - np.sqrt(matrix).sum(axis=0))
    if geometry != "minimum":
        gradient = costs - (context_scale * gradient_context if geometry == "context" else gradient_jensen)
        return {"normalized_global_gap_bound": max(0., float(gradient @ p - gradient.min())),
                "certificate_method": "convex_tangent_simplex"}
    if lambda_alpha > 0 and np.any(p <= 0):
        return {"normalized_global_gap_bound": math.inf}
    gradient_alpha = lambda_alpha * p ** (alpha - 1) / (1 - alpha) if lambda_alpha > 0 else np.zeros_like(p)
    value = bonuses(matrix, p, alpha)
    a, j = value["action_bonus"], context_scale * value["jensen_bonus"]
    positive = np.all(p > 0)
    gradient_action = 0.5 / np.sqrt(p) if positive else None

    def gap(weight):
        gradient = costs - gradient_alpha - 2 * gradient_jensen
        if weight != 0:
            if not positive:
                return math.inf
            gradient = gradient - 2 * weight * (gradient_action - gradient_jensen)
        return max(0., float(2 * (weight * a + (1 - weight) * j - min(a, j))
                             + gradient @ p - gradient.min()))

    candidates = [(gap(0.), 0.)]
    if positive:
        from scipy.optimize import minimize_scalar
        for weight in (1., float(np.clip(branch_weight, 0, 1))):
            candidates.append((gap(weight), weight))
        solution = minimize_scalar(gap, bounds=(0., 1.), method="bounded", options={"xatol": 1e-13})
        candidates.append((gap(solution.x), float(solution.x)))
    bound, weight = min(candidates)
    return {"normalized_global_gap_bound": bound, "certificate_action_weight": weight,
            "certificate_method": "minimum_regularizer_dual_tangent"}


class _Base:
    def __init__(self, mechanism: Any, horizon: int, rng: np.random.Generator):
        self.mechanism, self.horizon, self.rng = mechanism, int(horizon), rng
        self.M = np.asarray(mechanism.M, dtype=float)
        self.n_contexts, self.n_actions = self.M.shape
        if self.horizon != horizon or self.horizon < 1 or self.n_actions < 1 or self.n_contexts < 1:
            raise ValueError("Positive horizon and mechanism dimensions required")
        if (self.n_actions != int(mechanism.n_actions)
                or self.n_contexts != int(mechanism.n_contexts)):
            raise ValueError("mechanism dimensions are inconsistent with M")
        if (not np.isfinite(self.M).all() or np.any(self.M < 0)
                or not np.allclose(self.M.sum(axis=0), 1, atol=1e-12, rtol=1e-12)):
            raise ValueError("M must have finite probability-distribution columns")

    @staticmethod
    def _round(t: int) -> int:
        if int(t) != t or t < 1:
            raise ValueError("Round t must be a positive integer")
        return int(t)


class TsallisINF(_Base):
    name = "Tsallis-INF"

    def __init__(self, mechanism, horizon, rng, estimator="rv", eta_scale=4.0, **options):
        super().__init__(mechanism, horizon, rng)
        if options:
            raise TypeError(f"Unknown {self.name} options: {sorted(options)}")
        if estimator not in {"rv", "iw"} or not np.isfinite(eta_scale) or eta_scale <= 0:
            raise ValueError("Valid estimator and positive finite eta_scale required")
        self.estimator, self.eta_scale = estimator, float(eta_scale)
        self.L, self._dual = np.zeros(self.n_actions), None

    def probabilities(self, t):
        eta = self.eta_scale / math.sqrt(self._round(t))
        p, self._dual = half_tsallis(self.L * eta / 4, self._dual)
        return p

    def update(self, t, action, context, loss, p):
        eta = self.eta_scale / math.sqrt(self._round(t))
        self.L += loss_estimate(action, loss, p, eta, self.estimator)
        self.L -= self.L.min()

    def diagnostics(self):
        return {"estimator": self.estimator, "eta_scale": self.eta_scale,
                "estimator_space": "action", "solver": "scalar KKT",
                "half_tsallis_dual": self._dual}


class _CausalBase(_Base):
    def __init__(self, mechanism, horizon, rng, estimator="rv", eta_scale=4.0,
                 baseline_mode="fine", **options):
        super().__init__(mechanism, horizon, rng)
        if options:
            raise TypeError(f"Unknown {self.name} options: {sorted(options)}")
        if estimator not in {"rv", "iw"} or not np.isfinite(eta_scale) or eta_scale <= 0:
            raise ValueError("Valid estimator and positive finite eta_scale required")
        if baseline_mode not in {"fine", "coarse"}:
            raise ValueError("baseline_mode must be 'fine' or 'coarse'")
        self.estimator, self.eta_scale, self.baseline_mode = estimator, float(eta_scale), baseline_mode
        self._groups = None
        if baseline_mode == "coarse":
            labels = getattr(mechanism, "baseline_groups", None)
            if labels is None or np.asarray(labels).shape != (self.n_contexts,):
                raise ValueError("Coarse RV requires mechanism.baseline_groups with one label per context")
            self._groups = np.asarray(labels)
        self.G = np.zeros(self.n_contexts)
        self._last = {}
        self._zero_reachable_count = 0

    def update(self, t, action, context, loss, p):
        eta = self.eta_scale / math.sqrt(self._round(t))
        q = self.M @ np.asarray(p)
        missing = int(np.count_nonzero((q <= 0) & np.any(self.M > 0, axis=1)))
        self._zero_reachable_count += missing
        if missing:
            raise NumericalSolverError("Causal update lost reachable context support; no estimator floor inserted")
        self.G += loss_estimate(context, loss, q, eta, self.estimator, self._groups)
        self.G -= self.G.min()

    def diagnostics(self):
        return {"estimator": self.estimator, "eta_scale": self.eta_scale,
                "baseline_mode": self.baseline_mode, "estimator_space": "context",
                "zero_reachable_contexts_across_updates": self._zero_reachable_count,
                **self._last}


class CTsallisAction(_CausalBase):
    name = "CTsallis-Action"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dual = None

    def probabilities(self, t):
        eta = self.eta_scale / math.sqrt(self._round(t))
        p, self._dual = half_tsallis((self.M.T @ self.G) * eta / 4, self._dual)
        self._last = {"solver": "scalar KKT", "half_tsallis_dual": self._dual}
        return p


class _ConicCausal(_CausalBase):
    """A reusable DPP conic program with only its loss vector changing."""
    geometry = "context"

    def __init__(self, mechanism, horizon, rng, context_bonus_scale=1.0,
                 solver_tolerance=1e-8, solver_max_iterations=300,
                 cleanup_tolerance=1e-7, solver_retry_tolerances=(1e-7,),
                 solver_gap_certificate_tolerance=None, solver_cold_retry=False,
                 solver_dual_certificate=False, solver_policy_polish=False, solver_trust_polish=False, **options):
        super().__init__(mechanism, horizon, rng, **options)
        self.context_bonus_scale = float(context_bonus_scale)
        self.solver_tolerance = float(solver_tolerance)
        self.cleanup_tolerance = float(cleanup_tolerance)
        self.solver_max_iterations = int(solver_max_iterations)
        self.solver_retry_tolerances = tuple(float(x) for x in solver_retry_tolerances)
        self.solver_gap_certificate_tolerance = (None if solver_gap_certificate_tolerance is None
                                                 else float(solver_gap_certificate_tolerance))
        self.solver_cold_retry = bool(solver_cold_retry)
        self.solver_dual_certificate = bool(solver_dual_certificate)
        self.solver_policy_polish = bool(solver_policy_polish)
        self._polished_count = 0
        self.solver_trust_polish = bool(solver_trust_polish)
        self._trust_polished_count = 0
        if self.solver_gap_certificate_tolerance is not None and not 0 < self.solver_gap_certificate_tolerance < 1:
            raise ValueError("solver_gap_certificate_tolerance must be None or a positive tolerance below one")
        if (not np.isfinite(self.context_bonus_scale) or self.context_bonus_scale <= 0
                or not 0 < self.solver_tolerance < 1 or not 0 < self.cleanup_tolerance < 1
                or self.solver_max_iterations < 1
                or any(not 0 < x < 1 for x in self.solver_retry_tolerances)):
            raise ValueError("Positive context bonus and valid solver tolerances required")
        self._problem = None
        self._cleanup_count = 0
        self._maximum_cleanup_l1 = 0.0
        self._solves = 0
        self._retry_count = 0
        self._maximum_accepted_tolerance = 0.0
        self._gap_certified_count = 0
        self._maximum_certified_gap = 0.0

    def _build_problem(self):
        # Lazy import permits reading configuration/manifest without a solver.
        import cvxpy as cp

        self._p = cp.Variable(self.n_actions)
        self._cost = cp.Parameter(self.n_actions)
        constraints = [self._p >= 0, cp.sum(self._p) == 1]
        reachable = np.any(self.M > 0, axis=1)
        context = cp.sum(cp.sqrt(self.M[reachable] @ self._p))
        jensen = context - np.sqrt(self.M).sum(axis=0) @ self._p
        self._branch_constraints = None
        if self.geometry == "context":
            reward = self.context_bonus_scale * context
        elif self.geometry == "jensen_only":
            reward = self.context_bonus_scale * jensen
        else:
            self._u = cp.Variable()
            action = cp.sum(cp.sqrt(self._p)) - 1
            self._branch_constraints = [self._u <= action,
                                        self._u <= self.context_bonus_scale * jensen]
            constraints += self._branch_constraints
            reward = 2 * self._u
            if self.lambda_alpha > 0:
                # A small rational exponent is represented by conic lifts.
                # Reject unfaithful approximation instead of silently changing alpha.
                alpha_rational = Fraction(self.alpha).limit_denominator(10000)
                if abs(float(alpha_rational) - self.alpha) > 1e-12:
                    raise ValueError("Conic alpha must be representable within 1e-12 by a rational with denominator <=10000")
                reward += self.lambda_alpha * (cp.sum(cp.power(self._p, alpha_rational, max_denom=10000)) - 1) / (
                    self.alpha * (1 - self.alpha))
        self._problem = cp.Problem(cp.Minimize(self._cost @ self._p - reward), constraints)
        if not self._problem.is_dcp(dpp=True):
            raise RuntimeError("FTRL objective did not pass CVXPY DCP/DPP verification")

    def _coefficient(self, t):
        return 4 * math.sqrt(t) / self.eta_scale

    def _conic_dual_certificate(self, p):
        """Bound the objective gap using a cone-feasible dual and residual correction.

        For min c.x with Ax+s=b, z in K* gives a lower bound
        -b.z - sum(B*abs(c+A.T@z)) whenever an optimum has |x|<=B.
        The SOC lifts here have such a representation: probabilities, square
        roots and rational-power geometric means are bounded by one; u lies
        between zero and sqrt(A)-1. We use symmetric bounds conservatively.
        This optional fallback is limited to the verified alpha=2/3 lift.
        """
        import cvxpy as cp
        if self.geometry == 'minimum' and self.lambda_alpha > 0 and abs(self.alpha - 2/3) > 1e-14:
            return {'normalized_global_gap_bound': math.inf}
        data = self._problem.get_problem_data(cp.CLARABEL)[0]
        dims = data['dims']
        if dims.exp or dims.psd or dims.p3d:
            return {'normalized_global_gap_bound': math.inf}
        try:
            z = np.asarray(self._problem._solver_cache['CLARABEL'].get_solution().z).copy()
        except (KeyError, AttributeError):
            return {'normalized_global_gap_bound': math.inf}
        if not np.isfinite(z).all():
            return {'normalized_global_gap_bound': math.inf}
        position = dims.zero
        z[position:position+dims.nonneg] = np.maximum(z[position:position+dims.nonneg], 0)
        position += dims.nonneg
        for size in dims.soc:
            block = z[position:position+size]
            norm, first = np.linalg.norm(block[1:]), float(block[0])
            if norm <= -first:
                block[:] = 0
            elif norm > first:
                block[0] = (norm + first) / 2
                block[1:] *= block[0] / norm
            block[0] = max(block[0], np.linalg.norm(block[1:])) + 64*np.finfo(float).eps*max(1., np.linalg.norm(block))
            position += size
        if position != len(z):
            return {'normalized_global_gap_bound': math.inf}
        bounds = np.full(len(data['c']), max(1., math.sqrt(float(self.M.max()))) * (1+1e-12))
        offset = 0.
        if self.geometry == 'minimum':
            bounds[data['param_prob'].var_id_to_col[self._u.id]] = max(1., math.sqrt(self.n_actions)-1) * (1+1e-12)
            offset = self.lambda_alpha / (self.alpha * (1-self.alpha))
        residual = data['c'] + data['A'].T @ z
        lower = float(-data['b'] @ z - bounds @ np.abs(residual) + offset)
        value = bonuses(self.M, p, getattr(self, 'alpha', 2/3))
        reward = self.context_bonus_scale * value['raw_context_bonus' if self.geometry == 'context' else 'jensen_bonus']
        if self.geometry == 'minimum':
            reward = 2*min(value['action_bonus'], self.context_bonus_scale*value['jensen_bonus']) + self.lambda_alpha*value['alpha_bonus']
        upper = float(self._cost.value @ p - reward)
        # Cushion floating-point dot products in the reported upper bound.
        cushion = 128*np.finfo(float).eps*(1 + abs(upper) + abs(offset) + np.abs(data['b']) @ np.abs(z) + bounds @ np.abs(residual))
        return {'normalized_global_gap_bound': max(0., upper-lower+float(cushion)),
                'certificate_method': 'cone_dual_with_bounded_residual',
                'dual_stationarity_residual_max': float(np.max(np.abs(residual)))}

    def _polish_candidate(self, initial):
        """Refine the same objective; only a global gap check can accept it.

        Softmax coordinates supply feasible trial policies without a final
        probability floor. A damped Newton step resolves tiny probabilities
        in the alpha-stabilized smooth branch. Neither optimizer's success
        flag is used as evidence of optimality.
        """
        from scipy.optimize import minimize
        from scipy.special import softmax
        alpha = getattr(self, 'alpha', 2/3)
        lam = getattr(self, 'lambda_alpha', 0.)
        mat = self.M[np.any(self.M > 0, axis=1)]
        h = np.sqrt(self.M).sum(axis=0)

        def objective_gradient(p):
            q = mat @ p
            if np.any(p <= 0) or np.any(q <= 0):
                return math.inf, np.full_like(p, math.nan), 0.
            value = bonuses(self.M, p, alpha)
            context_gradient = mat.T @ (.5 / np.sqrt(q))
            if self.geometry != 'minimum':
                reward = self.context_bonus_scale * value['raw_context_bonus' if self.geometry == 'context' else 'jensen_bonus']
                gradient = self._cost.value - self.context_bonus_scale * (context_gradient - (h if self.geometry == 'jensen_only' else 0))
                weight = 0.
            else:
                weight = float(value['action_bonus'] < self.context_bonus_scale * value['jensen_bonus'])
                reward = 2*min(value['action_bonus'], self.context_bonus_scale*value['jensen_bonus']) + lam*value['alpha_bonus']
                gradient = self._cost.value - weight/np.sqrt(p) - 2*(1-weight)*self.context_bonus_scale*(context_gradient-h)
                if lam:
                    gradient -= lam*p**(alpha-1)/(1-alpha)
            return float(self._cost.value @ p - reward), gradient, weight

        def certificate(p):
            tangent = convex_gap_certificate(self.M, p, self._cost.value, self.geometry,
                                               self.context_bonus_scale, alpha, lam)
            dual = self._conic_dual_certificate(p) if self.solver_dual_certificate else {'normalized_global_gap_bound': math.inf}
            return min((tangent, dual), key=lambda c: c['normalized_global_gap_bound'])

        def log_objective(logp):
            p = softmax(logp)
            value, gradient, _ = objective_gradient(p)
            return value, p*(gradient-gradient @ p)

        iterations = 0
        for mixing in (1e-14, 1e-8):
            start = (initial + mixing) / (1 + len(initial)*mixing)
            result = minimize(log_objective, np.log(start), jac=True, method='L-BFGS-B',
                              options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 1000, 'maxls': 50})
            p = softmax(result.x)
            iterations += result.nit
            check = certificate(p)
            if check['normalized_global_gap_bound'] <= self.solver_gap_certificate_tolerance:
                return p, dict(check, policy_polish_iterations=iterations)
            if not (self.geometry == 'minimum' and lam > 0):
                continue
            for _ in range(100):
                value, gradient, weight = objective_gradient(p)
                q = mat @ p
                hessian = (1-weight)*self.context_bonus_scale*.5*(mat.T*q**-1.5) @ mat
                hessian += np.diag(.5*weight*p**-1.5 + lam*alpha*p**(alpha-2))
                scaling = 1/np.sqrt(np.diag(hessian))
                try:
                    inverse = np.linalg.solve(hessian*scaling[:,None]*scaling[None,:],
                                              np.stack([gradient*scaling, scaling], axis=1))*scaling[:,None]
                except np.linalg.LinAlgError:
                    break
                direction = -inverse[:,0] + inverse[:,1]*(inverse[:,0].sum()/inverse[:,1].sum())
                step = min(1., min(-.99*p[direction < 0]/direction[direction < 0], default=1.))
                for _ in range(60):
                    trial = p + step*direction
                    trial /= trial.sum()
                    if objective_gradient(trial)[0] <= value + 1e-4*step*(gradient @ direction) + 1e-14:
                        break
                    step /= 2
                else:
                    break
                p = trial
                iterations += 1
                check = certificate(p)
                if check['normalized_global_gap_bound'] <= self.solver_gap_certificate_tolerance:
                    return p, dict(check, policy_polish_iterations=iterations)
        raise NumericalSolverError('Policy polishing did not reach the global objective-gap tolerance')

    def _trust_polish_candidate(self, initial):
        """Interior constrained refinement, accepted only by the original gap test."""
        from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
        n = self.n_actions
        minimum = self.geometry == 'minimum'
        alpha = getattr(self, 'alpha', 2/3)
        lam = getattr(self, 'lambda_alpha', 0.)
        mat = self.M[np.any(self.M > 0, axis=1)]
        h = np.sqrt(self.M).sum(axis=0)
        size = n + int(minimum)

        def objective(x):
            p = x[:n]
            value = bonuses(self.M, p, alpha)
            reward = (2*x[-1] + lam*value['alpha_bonus'] if minimum else
                      self.context_bonus_scale*value['raw_context_bonus' if self.geometry == 'context' else 'jensen_bonus'])
            return float(self._cost.value @ p - reward)

        def gradient(x):
            p = x[:n]
            if minimum:
                g = self._cost.value - lam*p**(alpha-1)/(1-alpha) if lam else self._cost.value.copy()
                return np.r_[g, -2.]
            return self._cost.value - self.context_bonus_scale*(mat.T @ (.5/np.sqrt(mat @ p)) - (h if self.geometry == 'jensen_only' else 0))

        def hessian(x):
            p = x[:n]
            if minimum:
                return np.diag(np.r_[lam*alpha*p**(alpha-2), 0.]) if lam else np.zeros((size,size))
            return .25*self.context_bonus_scale*(mat.T*(mat @ p)**-1.5) @ mat

        def branch_values(x):
            value = bonuses(self.M, x[:n], alpha)
            return np.array([value['action_bonus']-x[-1], self.context_bonus_scale*value['jensen_bonus']-x[-1]])

        def branch_jacobian(x):
            p = x[:n]
            return np.stack([np.r_[.5/np.sqrt(p), -1.],
                             np.r_[self.context_bonus_scale*(mat.T @ (.5/np.sqrt(mat @ p))-h), -1.]])

        def branch_hessian(x, weights):
            p = x[:n]
            result = np.zeros((size,size))
            result[:n,:n] = -.25*(weights[0]*np.diag(p**-1.5) + weights[1]*self.context_bonus_scale*(mat.T*(mat @ p)**-1.5) @ mat)
            return result

        start = (initial + 1e-12)/(1+n*1e-12)
        if minimum:
            value = bonuses(self.M, start, alpha)
            start = np.r_[start, min(value['action_bonus'],self.context_bonus_scale*value['jensen_bonus'])-1e-6]
        equality = np.r_[np.ones(n), 0.] if minimum else np.ones(n)
        constraints = [LinearConstraint(equality[None,:], 1., 1.)]
        if minimum:
            constraints.append(NonlinearConstraint(branch_values, 0., np.inf, jac=branch_jacobian, hess=branch_hessian))
        lower = np.r_[np.zeros(n), -np.inf] if minimum else np.zeros(n)
        iterations = 0
        for barrier in (1e-8, 1e-9, 1e-10):
            try:
                result = minimize(objective, start, jac=gradient, hess=hessian, method='trust-constr',
                                  bounds=Bounds(lower, np.full(size,np.inf), keep_feasible=True), constraints=constraints,
                                  options={'gtol':1e-11, 'xtol':1e-14, 'barrier_tol':1e-12,
                                           'initial_barrier_parameter':barrier,
                                           'initial_barrier_tolerance':barrier/100, 'maxiter':1000})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            iterations += result.nit
            p = result.x[:n].copy()
            if not np.isfinite(p).all() or np.any(p <= 0):
                continue
            p /= p.sum()
            checks = [convex_gap_certificate(self.M, p, self._cost.value, self.geometry,
                                              self.context_bonus_scale, alpha, lam)]
            if self.solver_dual_certificate:
                checks.append(self._conic_dual_certificate(p))
            certificate = min(checks, key=lambda c:c['normalized_global_gap_bound'])
            if certificate['normalized_global_gap_bound'] <= self.solver_gap_certificate_tolerance:
                return p, dict(certificate, policy_polish_iterations=iterations,
                               policy_polish_method='trust-constr', trust_barrier_parameter=barrier)
        raise NumericalSolverError('Interior constrained polishing did not reach the global objective-gap tolerance')

    def _verified_candidate(self, tolerance):
        """Check a solver candidate before it may become a sampling policy."""
        raw = np.asarray(self._p.value, dtype=float).reshape(-1)
        if (not np.isfinite(raw).all() or raw.min() < -self.cleanup_tolerance
                or abs(raw.sum() - 1) > self.cleanup_tolerance):
            raise NumericalSolverError("invalid conic simplex solution")
        p = np.maximum(raw, 0)
        p[np.argmax(p)] += 1 - p.sum()
        cleanup = float(np.abs(p - raw).sum())
        if cleanup > self.cleanup_tolerance:
            raise NumericalSolverError("simplex cleanup exceeded tolerance")
        q = self.M @ p
        if np.any((q <= 0) & np.any(self.M > 0, axis=1)):
            raise NumericalSolverError("reachable context support lost to numerical underflow; no floor inserted")
        if self.geometry == "minimum" and self.lambda_alpha > 0 and np.any(p <= 0):
            raise NumericalSolverError("alpha-stabilized action support lost to numerical underflow; no floor inserted")
        values = bonuses(self.M, p, getattr(self, "alpha", 2 / 3))
        scaled_jensen = self.context_bonus_scale * values["jensen_bonus"]
        detail = {"solver": "CVXPY/CLARABEL", "solver_status": self._problem.status,
                      "simplex_primal_residual": max(abs(float(raw.sum()) - 1), max(0.0, -float(raw.min()))),
                      "solver_iterations": int(self._problem.solver_stats.num_iters),
                      "solver_objective": float(self._problem.value),
                      "solver_tolerance": self.solver_tolerance,
                      "accepted_solver_tolerance": tolerance,
                      "cleanup_tolerance": self.cleanup_tolerance,
                      "minimum_action_probability": float(p.min()),
                      "minimum_reachable_context_probability": float(q[np.any(self.M > 0, axis=1)].min()),
                      "context_bonus_scale": self.context_bonus_scale,
                      "scaled_jensen_bonus": scaled_jensen, **values}
        if self._branch_constraints is not None:
            duals = np.array([float(c.dual_value) for c in self._branch_constraints])
            dual_error = abs(float(duals.sum()) - 2)
            if (duals.min() < -self.cleanup_tolerance
                    or dual_error > 100 * self.cleanup_tolerance):
                raise NumericalSolverError("Minimum-regularizer duals failed stationarity certification")
            branch_slacks = np.array([values["action_bonus"], scaled_jensen]) - float(self._u.value)
            branch_violation = max(0.0, -float(branch_slacks.min()))
            complementarity = float(np.max(np.abs(duals * branch_slacks)))
            # Cleaning a negative residual at a genuine boundary affects sqrt
            # terms by O(sqrt(cleanup)). Report the feasible-policy certificate
            # separately from the solver's conic gap criterion.
            certificate_tolerance = 10 * math.sqrt(self.cleanup_tolerance) * max(
                1, self.context_bonus_scale * math.sqrt(self.n_contexts))
            if max(branch_violation, complementarity) > certificate_tolerance:
                raise NumericalSolverError("Minimum epigraph failed feasible-policy complementarity check")
            detail.update({"action_branch_weight": float(duals[0] / duals.sum()),
                               "jensen_action_weight": float(duals[0] / duals.sum()),
                               "context_branch_weight": float(duals[1] / duals.sum()),
                               "branch_dual_stationarity_error": dual_error,
                               "branch_primal_violation": branch_violation,
                               "branch_complementarity_residual": complementarity,
                               "min_branch": ("intersection" if abs(values["action_bonus"] - scaled_jensen) <= 10 * tolerance
                                              else "action" if values["action_bonus"] < scaled_jensen else "context")})
        return p, cleanup, detail

    def probabilities(self, t):
        self._round(t)
        if self.n_actions == 1:
            return np.ones(1)
        if self._problem is None:
            self._build_problem()
        import cvxpy as cp

        costs = self.M.T @ self.G
        self._cost.value = (costs - costs.min()) / self._coefficient(t)
        tolerances = tuple(dict.fromkeys((self.solver_tolerance, *self.solver_retry_tolerances)))
        strategies = [(tolerance, None) for tolerance in tolerances]
        if self.solver_cold_retry:
            strategies += [(1e-9, {}), (1e-9, {"static_regularization_constant": 1e-10}),
                           (1e-10, {"static_regularization_constant": 1e-12})]
        legacy_attempt_count = len(strategies)
        if self.solver_trust_polish:
            strategies.append((1e-10, {}))
        attempts = []
        for attempt, (tolerance, cold_options) in enumerate(strategies):
            trust_recovery = self.solver_trust_polish and attempt == legacy_attempt_count
            self._solves += 1
            self._retry_count += int(attempt > 0)
            record = {"tolerance": tolerance, "status": "solver_error", "warnings": [],
                      "cold_rebuild": cold_options is not None, "solver_options": cold_options or {},
                      "trust_polish_recovery": trust_recovery}
            try:
                if cold_options is not None:
                    # Rebuild only the numerical optimization problem. G, the
                    # estimator, and every algorithm parameter stay unchanged.
                    self._build_problem()
                    self._cost.value = (costs - costs.min()) / self._coefficient(t)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", UserWarning)
                    self._problem.solve(solver="CLARABEL", warm_start=True,
                                        tol_gap_abs=tolerance, tol_gap_rel=tolerance,
                                        tol_feas=tolerance, max_iter=self.solver_max_iterations,
                                        **(cold_options or {}))
                record["warnings"] = [str(w.message) for w in caught]
                record["status"] = self._problem.status
                independently_check = (self._problem.status == cp.OPTIMAL_INACCURATE
                                       and self.solver_gap_certificate_tolerance is not None
                                       and self._p.value is not None)
                if self._problem.status != cp.OPTIMAL and not independently_check:
                    raise NumericalSolverError(f"solver status {self._problem.status}")
                p, cleanup, detail = self._verified_candidate(tolerance)
                if independently_check:
                    certificate = convex_gap_certificate(
                        self.M, p, self._cost.value, self.geometry, self.context_bonus_scale,
                        getattr(self, "alpha", 2 / 3), getattr(self, "lambda_alpha", 0.),
                        detail.get("jensen_action_weight", .5))
                    bound = certificate["normalized_global_gap_bound"]
                    if self.solver_dual_certificate and bound > self.solver_gap_certificate_tolerance:
                        dual_certificate = self._conic_dual_certificate(p)
                        if dual_certificate['normalized_global_gap_bound'] < bound:
                            certificate = dual_certificate
                            bound = certificate['normalized_global_gap_bound']
                    if (self.solver_policy_polish or trust_recovery) and bound > self.solver_gap_certificate_tolerance:
                        polished, certificate = (self._trust_polish_candidate(p) if trust_recovery else self._polish_candidate(p))
                        self._p.value = polished
                        p, cleanup, detail = self._verified_candidate(tolerance)
                        polish_detail = {k:v for k,v in certificate.items() if k.startswith('policy_polish_') or k == 'trust_barrier_parameter'}
                        certificates = [convex_gap_certificate(
                            self.M, p, self._cost.value, self.geometry, self.context_bonus_scale,
                            getattr(self, 'alpha', 2/3), getattr(self, 'lambda_alpha', 0.))]
                        if self.solver_dual_certificate:
                            certificates.append(self._conic_dual_certificate(p))
                        certificate = min(certificates, key=lambda c: c['normalized_global_gap_bound'])
                        certificate.update(polish_detail)
                        bound = certificate['normalized_global_gap_bound']
                        self._polished_count += 1
                    if not np.isfinite(bound) or bound > self.solver_gap_certificate_tolerance:
                        raise NumericalSolverError(f"independent normalized objective-gap bound {bound:g} exceeds tolerance")
                    detail.update(certificate, acceptance="independent_convex_gap",
                                  unnormalized_global_gap_bound=bound * self._coefficient(t))
                    self._gap_certified_count += 1
                    self._trust_polished_count += int(certificate.get("policy_polish_method") == "trust-constr")
                    self._maximum_certified_gap = max(self._maximum_certified_gap, bound)
                else:
                    detail["acceptance"] = "solver_optimal"
                record["accepted"] = True
                attempts.append(record)
                break
            except (cp.error.SolverError, NumericalSolverError) as exc:
                record.update({"accepted": False, "failure": str(exc)})
                attempts.append(record)
        else:
            self._last = {"solver": "CVXPY/CLARABEL", "solver_status": "failed",
                          "solver_calls": self._solves, "solver_retries_total": self._retry_count,
                          "solver_retry_tolerances": list(self.solver_retry_tolerances),
                          "last_solver_attempts": attempts,
                          "failed_round": t, "failed_normalized_costs": self._cost.value.tolist(),
                          "failed_cumulative_context_estimates": self.G.tolist()}
            reasons = "; ".join(f"tol={x['tolerance']:g}: {x['failure']}" for x in attempts)
            raise NumericalSolverError(f"{self.name}: no verified conic solution at t={t} ({reasons}); no floor or replacement policy inserted")
        self._cleanup_count += int(cleanup > 0)
        self._maximum_cleanup_l1 = max(self._maximum_cleanup_l1, cleanup)
        self._maximum_accepted_tolerance = max(self._maximum_accepted_tolerance, tolerance)
        self._last = {**detail, "solver_calls": self._solves,
                      "gap_certified_decisions": self._gap_certified_count,
                      "maximum_certified_normalized_gap": self._maximum_certified_gap,
                      "solver_gap_certificate_tolerance": self.solver_gap_certificate_tolerance,
                      "solver_cold_retry": self.solver_cold_retry,
                      "solver_dual_certificate": self.solver_dual_certificate,
                      "solver_policy_polish": self.solver_policy_polish,
                      "polished_decisions": self._polished_count,
                      "solver_trust_polish": self.solver_trust_polish,
                      "trust_polished_decisions": self._trust_polished_count,
                      "solver_retries_total": self._retry_count,
                      "solver_retry_tolerances": list(self.solver_retry_tolerances),
                      "last_solver_attempts": attempts,
                      "maximum_accepted_solver_tolerance": self._maximum_accepted_tolerance,
                      "simplex_cleanup_count": self._cleanup_count,
                      "maximum_cleanup_l1": self._maximum_cleanup_l1}
        return p


class CTsallisContext(_ConicCausal):
    """Raw H_Z(Mp), including its affine preference for dispersed columns."""
    name = "CTsallis-Context"
    geometry = "context"


class CTsallisJensenOnly(_ConicCausal):
    """Affine-corrected context diagnostic with coefficient matched to Context."""
    name = "CTsallis-JensenOnly"
    geometry = "jensen_only"


class CTsallisJensen(_ConicCausal):
    name = "CTsallis-Jensen"
    geometry = "minimum"

    def __init__(self, mechanism, horizon, rng, alpha=2 / 3, lambda_alpha=None,
                 regularizer_scale=0.25, eta_scale=2.0, **options):
        self.alpha, self.regularizer_scale = float(alpha), float(regularizer_scale)
        if not 0.5 < self.alpha < 1 or not np.isfinite(regularizer_scale) or regularizer_scale <= 0:
            raise ValueError("alpha must be in (1/2,1); regularizer_scale positive and finite")
        super().__init__(mechanism, horizon, rng, eta_scale=eta_scale, **options)
        if self.n_actions == 1 or self.n_contexts == 1:
            calibrated = 0.0
        else:
            diameter = math.expm1((1 - self.alpha) * math.log(self.n_actions)) / (self.alpha * (1 - self.alpha))
            calibrated = min(math.sqrt(self.n_actions ** self.alpha / diameter),
                             (math.sqrt(self.n_contexts) - 1) / diameter)
        self.lambda_alpha = calibrated if lambda_alpha is None else float(lambda_alpha)
        if not np.isfinite(self.lambda_alpha) or self.lambda_alpha < 0:
            raise ValueError("lambda_alpha must be finite and nonnegative")

    def _coefficient(self, t):
        # eta controls RV threshold only, as in the requested calibration.
        return self.regularizer_scale * math.sqrt(t)

    def diagnostics(self):
        return {**super().diagnostics(), "alpha": self.alpha,
                "lambda_alpha": self.lambda_alpha, "regularizer_scale": self.regularizer_scale,
                "geometry": self.geometry}


class CTsallisJensenNoAlpha(CTsallisJensen):
    name = "CTsallis-Jensen-NoAlpha"

    def __init__(self, mechanism, horizon, rng, eta_scale=4.0, lambda_alpha=0.0, **options):
        if lambda_alpha != 0:
            raise ValueError("NoAlpha label requires lambda_alpha=0")
        super().__init__(mechanism, horizon, rng, eta_scale=eta_scale, lambda_alpha=0, **options)
