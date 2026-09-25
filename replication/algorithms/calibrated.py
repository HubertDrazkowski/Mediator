"""Explicit native objectives with independently specified c and RV cutoff.

No alpha is silently added to Action or raw Context. This module implements
the native-objective fairness option; exact geometry ablations must receive
separate labels and definitions rather than reusing the historical names.
"""
from __future__ import annotations

import math


from .base.tsallis import TsallisINF, CTsallisAction, half_tsallis
from .solvers.fast_tsallis import FastContext, FastJensen


class CalibratedINF(TsallisINF):
    def __init__(self,*args,c=1.,**kwargs):
        self.c=float(c)
        super().__init__(*args,**kwargs)

    def probabilities(self,t):
        p,self._dual=half_tsallis(self.L/(self.c*math.sqrt(self._round(t))),self._dual)
        return p


class CalibratedAction(CTsallisAction):
    def __init__(self,*args,c=1.,**kwargs):
        self.c=float(c)
        super().__init__(*args,**kwargs)

    def probabilities(self,t):
        p,self._dual=half_tsallis((self.M.T@self.G)/(self.c*math.sqrt(self._round(t))),self._dual)
        self._last={'solver':'scalar KKT','half_tsallis_dual':self._dual,'main_coefficient_c':self.c}
        return p


class CalibratedContext(FastContext):
    def __init__(self,*args,c=1.,**kwargs):
        self.c=float(c)
        super().__init__(*args,**kwargs)

    def _coefficient(self,t):
        return self.c*math.sqrt(self._round(t))


def instantiate(spec,mechanism,horizon,rng):
    c,rho,k=spec['c'],spec['rho'],spec['cutoff']
    if c<=0 or rho<0 or k<=0: raise ValueError('Invalid declared coefficients')
    options={'estimator':'rv','eta_scale':math.sqrt(k)}
    method=spec['method']
    if method in ('Tsallis-INF','CTsallis-Action','CTsallis-Context'):
        if rho!=0: raise ValueError('Native standalone objectives exclude alpha')
        cls={'Tsallis-INF':CalibratedINF,'CTsallis-Action':CalibratedAction,'CTsallis-Context':CalibratedContext}[method]
        return cls(mechanism,horizon,rng,c=c,**options)
    a,z=mechanism.n_actions,mechanism.n_contexts
    alpha=spec['alpha']
    diameter=math.expm1((1-alpha)*math.log(a))/(alpha*(1-alpha)) if a>1 else 0.
    calibrated=min(math.sqrt(a**alpha/diameter),(math.sqrt(z)-1)/diameter) if diameter>0 else 0.
    options.update(alpha=alpha,regularizer_scale=c/2,lambda_alpha=(2*rho/c)*calibrated)
    if method=='CTsallis-Jensen':
        if rho<=0: raise ValueError('Use the explicitly labelled rho0 method')
        return FastJensen(mechanism,horizon,rng,**options)
    if method=='CTsallis-Jensen-rho0':
        if rho!=0: raise ValueError('The rho0 ablation must have exactly zero alpha coefficient')
        from .noalpha_solver import FastNoAlphaJensen
        return FastNoAlphaJensen(mechanism,horizon,rng,**options)
    raise ValueError('Unknown method: '+method)
