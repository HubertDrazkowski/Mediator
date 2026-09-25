"""Certified minimum-regularizer solver when the alpha coefficient is zero.

The objective is unchanged: costs@p - 2 min(H_A(p), s J_M(p)). Action
probabilities may be zero, while every reachable context must retain strictly
positive probability. No probability floor enters the feasible set, returned
policy, or importance denominator.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base.tsallis import CTsallisContext, bonuses, half_tsallis
from .solvers.fast_solver import CandidateFailure, ContextNewton, MinimumNewton
from .solvers.fast_tsallis import FastJensen, _DirectConic


class NoAlphaDual:
    """Solve the smooth branch-mixture dual, including its boundary endpoint.

    At weight zero, f_0=(costs+2s h)@p-2s sum sqrt(Mp) is a raw Context
    objective with an affine cost shift and can have zero action weights.
    For every strictly positive weight, the action entropy supplies a
    positive diagonal Hessian and an interior optimizer. A safeguarded
    Newton search in the scalar branch weight resolves branch intersections.
    """
    def __init__(self, mechanism, horizon, rng, context_scale=1., tolerance=5e-7):
        self.M=np.asarray(mechanism.M,dtype=float)
        self.reachable=self.M[np.any(self.M>0,axis=1)]
        self.h=np.sqrt(self.M).sum(axis=0)
        self.context_scale=float(context_scale)
        self.beta=2.
        self.tolerance=float(tolerance)
        self.n=self.M.shape[1]
        self.context=ContextNewton(self.M,beta=self.beta*self.context_scale,
            tolerance=self.tolerance/20,max_iterations=100)
        self.context.warm=np.full(self.n,1/self.n)
        # The inherited smooth solver is valid for lambda=0 whenever weight>0.
        # Its constructor's positive-lambda guard is respected during setup;
        # weight-zero is handled separately, so no singular zero-lambda
        # Hessian is passed to it.
        self.smooth=MinimumNewton(self.M,beta=self.beta,lam=1.,context_scale=self.context_scale,
            tolerance=self.tolerance,max_iterations=100)
        self.smooth.lam=0.
        self.context_learner=CTsallisContext(mechanism,horizon,rng,
            context_bonus_scale=self.beta*self.context_scale)
        self.context_conic=None
        self.warm=None
        self.weight=0.
        self.counters={'pure_jensen_newton':0,'pure_jensen_conic':0,'action_endpoint':0,
                       'interior_branch_solves':0,'dual_iterations':0}

    def _bonuses(self,p):
        return float(np.sqrt(p).sum()-1),float(self.context_scale*(np.sqrt(self.reachable@p).sum()-self.h@p))

    def _support(self,p):
        return (np.isfinite(p).all() and np.all(p>=0) and abs(p.sum()-1)<1e-10
                and np.all(self.reachable@p>0))

    def _endpoint_zero(self,costs):
        shifted=costs+self.beta*self.context_scale*self.h
        try:
            p,it=self.context.solve(shifted)
            self.counters['pure_jensen_newton']+=1
            q=self.reachable@p
            grad=shifted-self.beta*self.context_scale*(self.reachable.T@(.5/np.sqrt(q)))
            gap=max(0.,float(grad@p-grad.min()))
            method='pure_jensen_full_simplex_tangent'
        except (CandidateFailure,np.linalg.LinAlgError,FloatingPointError,ValueError):
            if self.context_conic is None:
                self.context_conic=_DirectConic(self.context_learner)
            p,it,status=self.context_conic.solve(shifted)
            self.counters['pure_jensen_conic']+=1
            if not self._support(p):
                raise CandidateFailure('Pure Jensen endpoint lost reachable-context support')
            q=self.reachable@p
            grad=shifted-self.beta*self.context_scale*(self.reachable.T@(.5/np.sqrt(q)))
            tangent=max(0.,float(grad@p-grad.min()))
            dual=self.context_conic.certificate(self.context_learner,p,shifted)
            gap=min(tangent,dual)
            method='pure_jensen_conic_dual' if dual<tangent else 'pure_jensen_full_simplex_tangent'
        if not self._support(p) or not np.isfinite(gap) or gap>self.tolerance/2:
            raise CandidateFailure(f'Pure Jensen endpoint uncertified: {gap}')
        self.context.warm=p.copy()
        a,j=self._bonuses(p)
        cushion=64*np.finfo(float).eps*(1+np.abs(costs).max()+abs(a)+abs(j))
        fullgap=gap+self.beta*max(0.,j-a)+cushion
        return p,float(fullgap),method

    def _dual_curvature(self,p,weight):
        q=self.reachable@p
        v=.5/np.sqrt(p)-self.context_scale*(self.reachable.T@(.5/np.sqrt(q))-self.h)
        hessian=self.beta*(1-weight)*self.context_scale*.25*((self.reachable.T*q**-1.5)@self.reachable)
        hessian.flat[::self.n+1]+=self.beta*weight*.25*p**-1.5
        scale=1/np.sqrt(np.diag(hessian))
        inverse=cho_solve(cho_factor(hessian*scale[:,None]*scale[None,:],check_finite=False),
            np.stack((v*scale,scale),axis=1),check_finite=False)*scale[:,None]
        projected=inverse[:,0]-inverse[:,1]*(inverse[:,0].sum()/inverse[:,1].sum())
        return -self.beta**2*float(v@projected)

    def solve(self,costs):
        costs=np.asarray(costs,dtype=float)
        p0,gap0,method0=self._endpoint_zero(costs)
        if gap0<=self.tolerance:
            self.warm=p0.copy();self.weight=0.
            return p0,{'normalized_global_gap_bound':gap0,'certificate_action_weight':0.,
                       'certificate_method':method0,'noalpha_dual_iterations':0}
        p1,_=half_tsallis(costs/self.beta)
        gap1=self.smooth.certificate(p1,costs,1.)
        if gap1<=self.tolerance:
            self.warm=p1.copy();self.weight=1.;self.counters['action_endpoint']+=1
            return p1,{'normalized_global_gap_bound':gap1,'certificate_action_weight':1.,
                       'certificate_method':'action_endpoint_full_simplex_tangent','noalpha_dual_iterations':0}
        lower,upper=0.,1.
        weight=self.weight if 0<self.weight<1 else .5
        p=self.warm.copy() if self.warm is not None and np.all(self.warm>0) else (1-weight)*p0+weight*p1
        for outer in range(60):
            p=self.smooth._smooth(costs,weight,p)
            self.counters['interior_branch_solves']+=1
            gap=self.smooth.certificate(p,costs,weight)
            if gap<=self.tolerance:
                self.warm=p.copy();self.weight=weight;self.counters['dual_iterations']+=outer+1
                return p,{'normalized_global_gap_bound':gap,'certificate_action_weight':weight,
                           'certificate_method':'branch_mixture_full_simplex_tangent',
                           'noalpha_dual_iterations':outer+1}
            a,j=self._bonuses(p)
            slope=self.beta*(j-a)
            if slope>0:lower=weight
            else:upper=weight
            try:
                curvature=self._dual_curvature(p,weight)
                candidate=weight-slope/curvature if curvature<0 else math.nan
            except (np.linalg.LinAlgError,FloatingPointError):
                candidate=math.nan
            # A strict bracket is sufficient: a relative interior margin
            # would reject the tiny, useful Newton correction near the
            # previous round's root and turn warm starts into long bisections.
            weight=candidate if lower<candidate<upper else (lower+upper)/2
        raise CandidateFailure('No-alpha branch-weight search did not reach the global gap tolerance')


class FastJensenNoAlpha(FastJensen):
    """Drop-in Jensen constructor; lambda_alpha must equal zero."""
    name='CTsallis-Jensen-NoAlpha'
    def __init__(self,mechanism,horizon,rng,lambda_alpha=0.,**options):
        if lambda_alpha!=0:
            raise ValueError('FastJensenNoAlpha requires lambda_alpha=0')
        super().__init__(mechanism,horizon,rng,lambda_alpha=0.,**options)
        self._noalpha=NoAlphaDual(mechanism,horizon,rng,context_scale=self.context_bonus_scale,
                                 tolerance=self.fast_gap_tolerance/2)
        self._fast_counts['noalpha_dual']=0

    def probabilities(self,t):
        self._round(t)
        if self._dominant is not None or self.n_actions==1:
            return super().probabilities(t)
        costs=self.M.T@self.G
        costs=(costs-costs.min())/self._coefficient(t)
        try:
            p,certificate=self._noalpha.solve(costs)
            if not self._noalpha._support(p):
                raise CandidateFailure('No-alpha candidate violates simplex/context support')
            weight=certificate['certificate_action_weight']
            gap=min(self._certificate(p,costs,weight),certificate['normalized_global_gap_bound'])
            if not np.isfinite(gap) or gap>self.fast_gap_tolerance:
                raise CandidateFailure(f'No-alpha candidate did not certify: {gap}')
        except (CandidateFailure,np.linalg.LinAlgError,FloatingPointError,ValueError):
            self._fast_failures['noalpha_dual']=self._fast_failures.get('noalpha_dual',0)+1
            p=super().probabilities(t)
            self._noalpha.warm=p.copy()
            self._noalpha.weight=self._last.get('action_branch_weight',0.)
            self._last['noalpha_inner_counts']=dict(self._noalpha.counters)
            return p
        self._fast_counts['noalpha_dual']+=1
        self._fast_max_gap=max(self._fast_max_gap,gap)
        value=bonuses(self.M,p,self.alpha)
        self._last={'solver':'certified_noalpha_branch_dual','solver_status':'certified',
            'normalized_global_gap_bound':gap,'maximum_certified_normalized_gap':self._fast_max_gap,
            'unnormalized_global_gap_bound':gap*self._coefficient(t),
            'certificate_method':certificate['certificate_method'],
            'noalpha_dual_iterations':certificate['noalpha_dual_iterations'],
            'noalpha_inner_counts':dict(self._noalpha.counters),
            'fast_solver_counts':dict(self._fast_counts),'fast_solver_fallbacks':dict(self._fast_failures),
            'minimum_action_probability':float(p.min()),
            'minimum_reachable_context_probability':float((self._reachable_M@p).min()),
            'context_bonus_scale':self.context_bonus_scale,
            'jensen_action_weight':weight,'action_branch_weight':weight,'context_branch_weight':1-weight,**value}
        return p


# Public spelling used by the calibrated objective adapter.
FastNoAlphaJensen=FastJensenNoAlpha
