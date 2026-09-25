"""Isolated experimental optimizers; no learner integration or experiment execution.

Candidates are accepted only using a convex global objective-gap certificate.
No probability floor modifies the feasible set or returned policy.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class CandidateFailure(RuntimeError):
    pass


def dominant_groups(matrix):
    """Recognize M[:,a] = b + s e_group[a], including unrepresented contexts."""
    matrix = np.asarray(matrix, dtype=float)
    b = float(matrix.min())
    s = float(matrix.max() - b)
    z, a = matrix.shape
    if b <= 0 or s <= 0 or abs(z * b + s - 1) > 1e-12:
        return None
    labels = np.argmax(matrix, axis=0)
    expected = np.full_like(matrix, b)
    expected[labels, np.arange(a)] += s
    if not np.array_equal(matrix, expected):
        return None
    represented, groups, counts = np.unique(labels, return_inverse=True, return_counts=True)
    representatives = np.array([np.flatnonzero(groups == i)[0] for i in range(len(counts))])
    return b, s, groups, counts, representatives


def separable(costs, b, s, counts, beta=1., lam=0., alpha=2/3,
              warm=None, tolerance=1e-9, max_iterations=100):
    """Min c·w − beta sum sqrt(b+s w) − lam/(α(1−α)) sum k^(1−α)w^α.

    For dominant columns, uniform allocation within each duplicate group
    maximizes both action entropies while preserving context distribution.
    The raw-context and Jensen affine terms differ only by a constant here.
    Jensen <= sqrt(s)(sum sqrt(w)-1) <= sqrt(s) H_A(p), so with a context
    scale <=1/sqrt(s) the minimum regularizer is exactly the context branch.
    """
    costs = np.asarray(costs, dtype=float)
    n = len(costs)
    d = beta * s / 2
    if lam == 0:
        lower = float(np.max(d / np.sqrt(b+s) - costs))
        upper = float(np.max(d / np.sqrt(b+s/n) - costs))
        nu = (lower + upper)/2
        for iteration in range(max_iterations):
            q = (d/(costs+nu))**2
            p = np.maximum((q-b)/s, 0)
            residual = p.sum()-1
            if abs(residual) <= 1e-13:
                p[p.argmax()] -= residual
                return p, iteration+1
            if residual > 0:
                lower = nu
            else:
                upper = nu
            derivative = -2*np.sum(q[p > 0]/(costs[p > 0]+nu))/s
            candidate = nu-residual/derivative if derivative else np.nan
            nu = candidate if lower < candidate < upper else (lower+upper)/2
        raise CandidateFailure("Separable scalar root did not converge")
    weights = np.asarray(counts, dtype=float)**(1-alpha)
    coefficient = lam*weights/(alpha*(1-alpha))
    p = np.full(n, 1/n) if warm is None else np.asarray(warm).copy()
    if np.any(p <= 0):
        p = np.full(n, 1/n)
    def value(x):
        return costs@x - beta*np.sqrt(b+s*x).sum() - coefficient@x**alpha
    for iteration in range(max_iterations):
        q = b+s*p
        gradient = costs-d/np.sqrt(q)-lam*weights*p**(alpha-1)/(1-alpha)
        gap = float(gradient@p-gradient.min())
        if gap <= tolerance:
            return p, iteration+1
        hessian = .5*d*s/q**1.5 + lam*weights*p**(alpha-2)
        inverse = 1/hessian
        nu = (gradient*inverse).sum()/inverse.sum()
        direction = -(gradient-nu)*inverse
        step = min(1., np.min(-.99*p[direction < 0]/direction[direction < 0], initial=1.))
        initial = value(p)
        for _ in range(60):
            trial = p + step*direction
            trial /= trial.sum()
            if np.all(trial > 0) and value(trial) <= initial+1e-4*step*(gradient@direction)+1e-13:
                break
            step *= .5
        else:
            raise CandidateFailure("Separable line search failed")
        p = trial
    raise CandidateFailure(f"Separable Newton gap {gap}")


class MinimumNewton:
    """Interior Newton plus the exact one-dimensional branch-mixture dual.

    Solves costs·p − beta min(H_A(p), s J_M(p)) − lam Hα(p).
    Positive lam guarantees an interior optimum. The lambda-zero general
    case is deliberately unsupported; use the conic solver for that case.
    """
    def __init__(self, matrix, beta=2., lam=1., alpha=2/3, context_scale=1.,
                 tolerance=1e-8, max_iterations=80):
        if lam <= 0:
            raise ValueError("Positive alpha coefficient required")
        matrix = np.asarray(matrix, dtype=float)
        self.M = matrix[np.any(matrix > 0, axis=1)]
        self.h = np.sqrt(matrix).sum(axis=0)
        self.n = matrix.shape[1]
        self.beta, self.lam, self.alpha = float(beta), float(lam), float(alpha)
        self.context_scale = float(context_scale)
        self.tolerance, self.max_iterations = float(tolerance), int(max_iterations)
        self.warm = np.full(self.n, 1/self.n)
        self.weight = 0.
        self.total_iterations = 0

    def _components(self, p):
        q = self.M@p
        a = np.sqrt(p).sum()-1
        j = self.context_scale*(np.sqrt(q).sum()-self.h@p)
        h_alpha = (np.sum(p**self.alpha)-1)/(self.alpha*(1-self.alpha))
        return q, a, j, h_alpha

    def _value(self, p, costs, w):
        _, a, j, ha = self._components(p)
        return costs@p-self.beta*(w*a+(1-w)*j)-self.lam*ha

    def certificate(self, p, costs, weight):
        q, a, j, ha = self._components(p)
        gradient = (costs-self.lam*p**(self.alpha-1)/(1-self.alpha)
                    -self.beta*(weight*.5/np.sqrt(p)+(1-weight)*self.context_scale*
                                (self.M.T@(.5/np.sqrt(q))-self.h)))
        gap = self.beta*(weight*a+(1-weight)*j-min(a,j))+gradient@p-gradient.min()
        cushion = 32*np.finfo(float).eps*(1+np.abs(gradient).max()+abs(float(costs@p)))
        return max(0., float(gap))+float(cushion)

    def _smooth(self, costs, weight, initial):
        p = initial.copy()
        for iteration in range(self.max_iterations):
            q, a, j, ha = self._components(p)
            gradient = (costs-self.lam*p**(self.alpha-1)/(1-self.alpha)
                        -self.beta*(weight*.5/np.sqrt(p)+(1-weight)*self.context_scale*
                                    (self.M.T@(.5/np.sqrt(q))-self.h)))
            if gradient@p-gradient.min() <= self.tolerance/20:
                self.total_iterations += iteration+1
                return p
            diagonal = self.beta*weight*.25*p**-1.5 + self.lam*p**(self.alpha-2)
            hessian = (self.beta*(1-weight)*self.context_scale*.25)*((self.M.T*q**-1.5)@self.M)
            hessian.flat[::self.n+1] += diagonal
            scale = 1/np.sqrt(np.diag(hessian))
            rhs = np.stack((gradient*scale, scale), axis=1)
            fact = cho_factor(hessian*scale[:,None]*scale[None,:], check_finite=False)
            inverse = cho_solve(fact, rhs, check_finite=False)*scale[:,None]
            direction = -inverse[:,0]+inverse[:,1]*(inverse[:,0].sum()/inverse[:,1].sum())
            step = min(1., np.min(-.99*p[direction < 0]/direction[direction < 0], initial=1.))
            initial_value = float(costs@p-self.beta*(weight*a+(1-weight)*j)-self.lam*ha)
            for _ in range(60):
                trial = p+step*direction
                trial /= trial.sum()
                if np.all(trial > 0) and self._value(trial, costs, weight) <= initial_value+1e-4*step*(gradient@direction)+1e-13:
                    break
                step *= .5
            else:
                raise CandidateFailure("General Newton line search failed")
            p = trial
        raise CandidateFailure("General Newton did not converge")

    def solve(self, costs):
        costs = np.asarray(costs, dtype=float)
        low, high = 0., 1.
        candidates = [self.weight]
        if self.weight != 0:
            candidates.append(0.)
        if self.weight != 1:
            candidates.append(1.)
        p = self.warm.copy()
        for iteration in range(60):
            weight = candidates[iteration] if iteration < len(candidates) else (low+high)/2
            p = self._smooth(costs, weight, p)
            certificate = self.certificate(p, costs, weight)
            if certificate <= self.tolerance:
                self.weight, self.warm = weight, p.copy()
                return p, {"normalized_global_gap_bound": certificate,
                           "certificate_action_weight": weight,
                           "iterations": self.total_iterations}
            _, a, j, _ = self._components(p)
            # d/du min_p f_u = beta*(J-A), a decreasing function of u.
            if j > a:
                low = max(low, weight)
            else:
                high = min(high, weight)
        raise CandidateFailure("Branch dual search did not converge")


class ContextNewton:
    """Working-set Newton for the raw-context objective on the full simplex.

    Working-set pruning is only a trial initialization, never a probability
    floor or a change to the feasible domain. The full-simplex tangent gap
    decides acceptance; inactive actions with a negative reduced gradient
    are admitted to the working set.
    """
    def __init__(self,matrix,beta=1.,tolerance=1e-7,max_iterations=40):
        self.M = np.asarray(matrix)[np.any(matrix>0,axis=1)]
        self.beta,self.tolerance,self.max_iterations=beta,tolerance,max_iterations
        self.warm = None

    def solve(self,costs):
        if self.warm is None:
            raise CandidateFailure('Context working set requires an initial conic policy')
        p=self.warm.copy()
        p[p<1e-9]=0
        p/=p.sum()
        if np.any(self.M@p<=0):
            p=self.warm.copy()
        for iteration in range(self.max_iterations):
            q=self.M@p
            gradient=costs-self.beta*self.M.T@(.5/np.sqrt(q))
            gap=float(gradient@p-gradient.min())
            if gap<=self.tolerance:
                self.warm=p.copy()
                return p,iteration+1
            active=np.flatnonzero(p>0)
            candidate=int(gradient.argmin())
            # First settle stationarity within the working set. Admission
            # before that can produce a direction outside the active face.
            within=float(np.max(gradient[active])-np.min(gradient[active]))
            if candidate not in active and within<max(1e-6,gap*.1):
                active=np.r_[active,candidate]
            mat=self.M[:,active]
            hessian=.25*self.beta*((mat.T*q**-1.5)@mat)
            scale=1/np.sqrt(np.maximum(np.diag(hessian),1e-300))
            scaled=hessian*scale[:,None]*scale[None,:]
            # Positive damping regularizes the search direction only.
            scaled.flat[::len(active)+1]+=1e-10
            inverse=cho_solve(cho_factor(scaled,check_finite=False),
                np.stack((gradient[active]*scale,scale),axis=1),check_finite=False)*scale[:,None]
            local=-inverse[:,0]+inverse[:,1]*(inverse[:,0].sum()/inverse[:,1].sum())
            direction=np.zeros_like(p)
            direction[active]=local
            if gradient@direction >= -1e-20 or np.any((p==0)&(direction<0)):
                # Feasible pairwise descent also admits a new minimum-
                # gradient action when the working Hessian is singular.
                away=active[np.argmax(gradient[active])]
                direction[:]=0
                direction[candidate]=1
                direction[away]-=1
            negative=direction<0
            if not np.any(negative):
                raise CandidateFailure('Context has no feasible descent direction')
            limit=np.min(-p[negative]/direction[negative])
            step=min(1.,limit)
            value=float(costs@p-self.beta*np.sqrt(q).sum())
            for _ in range(50):
                trial=p+step*direction
                trial[np.abs(trial)<1e-15]=0
                trial/=trial.sum()
                qt=self.M@trial
                if (np.all(trial>=0) and np.all(qt>0) and
                    costs@trial-self.beta*np.sqrt(qt).sum()<=value+1e-4*step*(gradient@direction)+1e-13):
                    break
                step*=.5
            else:
                raise CandidateFailure('Context line search failed')
            p=trial
        raise CandidateFailure(f'Context working-set Newton gap={gap}')
