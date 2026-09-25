"""Certified optimization acceleration without changing the learner objective.

Factory-compatible alternatives for Context/Jensen. Exact dominant-column
reduction, warm Newton on the branch-mixture dual, and direct Clarabel with
a cached CVXPY canonicalization all solve the existing normalized objective.
Every accepted decision has a global normalized objective-gap bound <=1e-6.
"""
from __future__ import annotations

import math
import numpy as np

from ..base import factory as ordinary_factory
from ..base.tsallis import CTsallisContext, CTsallisJensen, bonuses, NumericalSolverError
from .fast_solver import CandidateFailure, MinimumNewton, ContextNewton, dominant_groups, separable


class _DirectConic:
    def __init__(self, learner):
        import cvxpy as cp
        import clarabel
        from scipy.sparse import csc_matrix
        if learner._problem is None:
            learner._build_problem()
        learner._cost.value = np.zeros(learner.n_actions)
        self.data = learner._problem.get_problem_data(cp.CLARABEL)[0]
        dims = self.data['dims']
        if dims.exp or dims.psd or dims.p3d:
            raise CandidateFailure('Unsupported cone type in direct backend')
        cones = []
        if dims.zero:
            cones.append(clarabel.ZeroConeT(dims.zero))
        if dims.nonneg:
            cones.append(clarabel.NonnegativeConeT(dims.nonneg))
        cones.extend(clarabel.SecondOrderConeT(size) for size in dims.soc)
        self.offset = self.data['param_prob'].var_id_to_col[learner._p.id]
        self.n = learner.n_actions
        self.bounds = np.full(len(self.data['c']),1+1e-12)
        if learner.geometry == 'minimum':
            offset_u = self.data['param_prob'].var_id_to_col[learner._u.id]
            self.bounds[offset_u] = max(1.,math.sqrt(self.n)-1)*(1+1e-12)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = learner.solver_max_iterations
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
        settings.max_threads = 1
        self.settings = settings
        self.solver = clarabel.DefaultSolver(self.data.get('P', csc_matrix((len(self.data['c']),)*2)),
                                             self.data['c'], self.data['A'], self.data['b'], cones, settings)

    def solve(self, costs):
        q = self.data['c'].copy()
        q[self.offset:self.offset+self.n] = costs
        self.solver.update(q=q)
        result = self.solver.solve()
        self.result = result
        self.q = q
        raw = np.asarray(result.x[self.offset:self.offset+self.n])
        if not np.isfinite(raw).all() or raw.min() < -1e-7 or abs(raw.sum()-1) > 1e-7:
            raise CandidateFailure('Invalid direct-conic primal policy')
        p = np.maximum(raw, 0)
        p[p.argmax()] += 1-p.sum()
        return p, int(result.iterations), str(result.status)

    def certificate(self, learner, p, costs):
        """Cone-feasible dual bound with a bounded stationarity correction.

        The cached canonical variables are action probabilities, square roots,
        and (for alpha=2/3) geometric mean lifts, each bounded by one. The
        minimum epigraph variable admits an optimum in [0,sqrt(A)-1].
        """
        if learner.geometry == 'minimum' and learner.lambda_alpha and abs(learner.alpha-2/3)>1e-14:
            return math.inf
        z = np.asarray(self.result.z).copy()
        dims = self.data['dims']
        position = dims.zero
        z[position:position+dims.nonneg] = np.maximum(z[position:position+dims.nonneg],0)
        position += dims.nonneg
        for size in dims.soc:
            block = z[position:position+size]
            norm,first = np.linalg.norm(block[1:]),float(block[0])
            if norm <= -first:
                block[:] = 0
            elif norm > first:
                block[0] = (norm+first)/2
                block[1:] *= block[0]/norm
            block[0] = max(block[0],np.linalg.norm(block[1:]))+64*np.finfo(float).eps*max(1.,np.linalg.norm(block))
            position += size
        residual = self.q+self.data['A'].T@z
        value = bonuses(learner.M,p,getattr(learner,'alpha',2/3))
        if learner.geometry == 'minimum':
            upper = costs@p-2*min(value['action_bonus'],learner.context_bonus_scale*value['jensen_bonus'])-learner.lambda_alpha*value['alpha_bonus']
            offset = learner.lambda_alpha/(learner.alpha*(1-learner.alpha))
        else:
            upper = costs@p-learner.context_bonus_scale*value['raw_context_bonus']
            offset = 0.
        correction = self.bounds@np.abs(residual)
        lower = -self.data['b']@z-correction+offset
        cushion = 128*np.finfo(float).eps*(1+abs(upper)+abs(offset)+np.abs(self.data['b'])@np.abs(z)+correction)
        return max(0.,float(upper-lower))+float(cushion)


class _FastMixin:
    def __init__(self, *args, fast_gap_tolerance=1e-6, **kwargs):
        kwargs.setdefault('solver_gap_certificate_tolerance',fast_gap_tolerance)
        kwargs.setdefault('solver_cold_retry',True)
        kwargs.setdefault('solver_dual_certificate',True)
        kwargs.setdefault('solver_policy_polish',True)
        kwargs.setdefault('solver_trust_polish',True)
        super().__init__(*args, **kwargs)
        if not 0 < fast_gap_tolerance <= 1e-6:
            raise ValueError('fast_gap_tolerance must be in (0,1e-6]')
        self.fast_gap_tolerance = float(fast_gap_tolerance)
        self._dominant = dominant_groups(self.M)
        if self._dominant is not None and self.geometry == 'minimum':
            if self.context_bonus_scale*math.sqrt(self._dominant[1]) > 1+1e-14:
                self._dominant = None
        self._fast_warm = None
        self._fast_newton = None
        self._context_newton = ContextNewton(self.M,beta=self.context_bonus_scale,tolerance=self.fast_gap_tolerance/2)
        self._fast_conic = None
        self._fast_counts = {'separable':0,'newton':0,'context_newton':0,'direct_conic':0,'legacy_fallback':0}
        self._fast_failures = {}
        self._fast_max_gap = 0.
        self._fast_h = np.sqrt(self.M).sum(axis=0)
        self._reachable_M = self.M[np.any(self.M > 0, axis=1)]

    def _certificate(self, p, costs, weight=0.):
        if not np.isfinite(p).all() or p.min() < 0 or abs(p.sum()-1)>1e-12:
            return math.inf
        q = self._reachable_M@p
        if np.any(q <= 0):
            return math.inf
        gradient = costs-self.context_bonus_scale*(self._reachable_M.T@(.5/np.sqrt(q)))
        excess = 0.
        if self.geometry == 'minimum':
            if self.lambda_alpha > 0 and np.any(p <= 0):
                return math.inf
            a = np.sqrt(p).sum()-1
            j = self.context_bonus_scale*(np.sqrt(q).sum()-self._fast_h@p)
            gj = self.context_bonus_scale*(self._reachable_M.T@(.5/np.sqrt(q))-self._fast_h)
            gradient = costs-2*(1-weight)*gj
            if weight:
                if np.any(p <= 0):
                    return math.inf
                gradient -= weight/np.sqrt(p)
            if self.lambda_alpha:
                gradient -= self.lambda_alpha*p**(self.alpha-1)/(1-self.alpha)
            excess = 2*(weight*a+(1-weight)*j-min(a,j))
        gap = max(0., float(excess+gradient@p-gradient.min()))
        cushion = 64*np.finfo(float).eps*(1+np.abs(gradient).max()+np.abs(costs).max())
        return gap+float(cushion)

    def probabilities(self, t):
        self._round(t)
        costs = self.M.T@self.G
        costs = (costs-costs.min())/self._coefficient(t)
        if self.n_actions == 1:
            return np.ones(1)
        methods = []
        if self._dominant is not None:
            methods.append('separable')
        if self.geometry == 'minimum' and self.lambda_alpha > 0:
            methods.append('newton')
        if self.geometry == 'context' and self._context_newton.warm is not None:
            methods.append('context_newton')
        methods.append('direct_conic')
        for method in methods:
            try:
                weight = 0.
                if method == 'separable':
                    b,s,groups,counts,representatives = self._dominant
                    grouped, iterations = separable(costs[representatives], b,s,counts,
                        beta=(2 if self.geometry == 'minimum' else 1)*self.context_bonus_scale,
                        lam=getattr(self,'lambda_alpha',0.),alpha=getattr(self,'alpha',2/3),
                        warm=self._fast_warm,tolerance=self.fast_gap_tolerance/20)
                    p = grouped[groups]/counts[groups]
                    self._fast_warm = grouped
                    status = 'certified'
                elif method == 'context_newton':
                    p,iterations=self._context_newton.solve(costs)
                    status='certified'
                elif method == 'newton':
                    if self._fast_newton is None:
                        self._fast_newton = MinimumNewton(self.M, lam=self.lambda_alpha, alpha=self.alpha,
                            context_scale=self.context_bonus_scale,tolerance=self.fast_gap_tolerance/2)
                    before = self._fast_newton.total_iterations
                    p, certificate = self._fast_newton.solve(costs)
                    iterations = self._fast_newton.total_iterations-before
                    weight = certificate['certificate_action_weight']
                    status = 'certified'
                else:
                    if self._fast_conic is None:
                        self._fast_conic = _DirectConic(self)
                    p,iterations,status = self._fast_conic.solve(costs)
                    if self.geometry == 'minimum':
                        # The generic certificate optimizes the branch dual;
                        # this path is infrequent for positive-alpha Jensen.
                        from ..base.tsallis import convex_gap_certificate
                        cert = convex_gap_certificate(self.M,p,costs,self.geometry,self.context_bonus_scale,
                                                       self.alpha,self.lambda_alpha)
                        weight = cert.get('certificate_action_weight',0.)
                p[p.argmax()] += 1-p.sum()
                if np.any(p<0) or np.any(self._reachable_M@p<=0) or (self.geometry=='minimum' and self.lambda_alpha>0 and np.any(p<=0)):
                    raise CandidateFailure('Candidate lost required estimator/alpha support')
                gap = self._certificate(p,costs,weight)
                if method == 'direct_conic' and gap > self.fast_gap_tolerance:
                    gap = min(gap,self._fast_conic.certificate(self,p,costs))
                if not gap <= self.fast_gap_tolerance:
                    raise CandidateFailure(f'Uncertified policy gap={gap}')
            except (CandidateFailure,FloatingPointError,ValueError,np.linalg.LinAlgError) as exc:
                self._fast_failures[method] = self._fast_failures.get(method,0)+1
                continue
            self._fast_counts[method] += 1
            if self.geometry=='context':
                self._context_newton.warm=p.copy()
            self._fast_max_gap = max(self._fast_max_gap,gap)
            value = bonuses(self.M,p,getattr(self,'alpha',2/3))
            self._last = {'solver':f'certified_fast_{method}','solver_status':status,
                'normalized_global_gap_bound':gap,'maximum_certified_normalized_gap':self._fast_max_gap,
                'unnormalized_global_gap_bound':gap*self._coefficient(t),'solver_iterations':iterations,
                'fast_solver_counts':dict(self._fast_counts),'fast_solver_fallbacks':dict(self._fast_failures),
                'minimum_action_probability':float(p.min()),
                'minimum_reachable_context_probability':float((self._reachable_M@p).min()),
                'context_bonus_scale':self.context_bonus_scale,**value}
            if self.geometry == 'minimum':
                self._last.update(jensen_action_weight=weight,action_branch_weight=weight,
                                  context_branch_weight=1-weight)
            return p
        p = super().probabilities(t)
        from ..base.tsallis import convex_gap_certificate
        certificate = convex_gap_certificate(self.M,p,costs,self.geometry,self.context_bonus_scale,
            getattr(self,'alpha',2/3),getattr(self,'lambda_alpha',0.),self._last.get('action_branch_weight',.5))
        weight = certificate.get('certificate_action_weight',0.)
        gap = self._certificate(p,costs,weight)
        # Existing conic dual bound can certify an ill-conditioned tangent.
        if gap > self.fast_gap_tolerance and self.solver_dual_certificate:
            gap = min(gap,self._conic_dual_certificate(p)['normalized_global_gap_bound'])
        if not gap <= self.fast_gap_tolerance:
            raise NumericalSolverError(f'All accelerated/legacy policies failed global certification: gap={gap}')
        self._fast_counts['legacy_fallback'] += 1
        self._fast_max_gap = max(self._fast_max_gap,gap)
        self._last.update(normalized_global_gap_bound=gap,maximum_certified_normalized_gap=self._fast_max_gap,
                          fast_solver_counts=dict(self._fast_counts),fast_solver_fallbacks=dict(self._fast_failures))
        return p


class FastContext(_FastMixin, CTsallisContext):
    pass


class FastJensen(_FastMixin, CTsallisJensen):
    pass


def factory(name, mechanism, horizon, rng, options=None):
    klass = {'CTsallis-Context':FastContext,'CTsallis-Jensen':FastJensen}.get(name)
    if klass is None:
        return ordinary_factory(name, mechanism, horizon, rng, options)
    return klass(mechanism,horizon,rng,**(options or {}))
