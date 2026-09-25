"""Additional mediator refinement after a certified solver failure.

The objective and acceptance tolerance remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .base.tsallis import NumericalSolverError, bonuses
from .solvers.fast_solver import ContextNewton, CandidateFailure

def rescue_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ContextFailureRefinement:
    def __init__(self,learner,expected_prefix_rounds=None,expected_prefix_sha256=None):
        self.learner=learner
        self.expected_prefix_rounds=expected_prefix_rounds
        self.expected_prefix_sha256=expected_prefix_sha256
        self.prefix_digest=hashlib.sha256()
        self.prefix_verified=False
        self.rescues=[]

    def __getattr__(self,name):return getattr(self.learner,name)

    def probabilities(self,t):
        try:
            p=self.learner.probabilities(t)
        except NumericalSolverError as original_error:
            if self.expected_prefix_rounds is not None and not self.prefix_verified:
                if t!=self.expected_prefix_rounds+1 or self.prefix_digest.hexdigest()!=self.expected_prefix_sha256:
                    raise NumericalSolverError('Numerical rescue refused: accepted prefix differs from exact historical replay') from original_error
                self.prefix_verified=True
            p=self._refine(t,original_error)
        if self.expected_prefix_rounds is not None and t<=self.expected_prefix_rounds:
            self.prefix_digest.update(np.asarray(p,dtype=float).tobytes())
        return p

    def _refine(self,t,original_error):
        learner=self.learner
        costs=learner.M.T@learner.G
        costs=(costs-costs.min())/learner._coefficient(t)
        threshold=min(1e-6,learner.fast_gap_tolerance)
        seeds=[]
        variable=getattr(learner,'_p',None)
        if variable is not None and variable.value is not None:
            raw=np.asarray(variable.value,dtype=float).reshape(-1)
            if (np.isfinite(raw).all() and raw.min()>=-learner.cleanup_tolerance
                    and abs(raw.sum()-1)<=learner.cleanup_tolerance):
                initial=np.maximum(raw,0);initial[initial.argmax()]+=1-initial.sum()
                seeds.append(('last_conic_candidate',initial))
        previous=getattr(learner._context_newton,'warm',None)
        if previous is not None:seeds.append(('previous_context_warm',np.asarray(previous).copy()))
        attempts=[]
        for label,initial in seeds:
            if initial.min()<0 or np.any(learner._reachable_M@initial<=0):
                attempts.append({'initial':label,'accepted':False,'reason':'initial context support invalid'})
                continue
            solver=ContextNewton(learner.M,beta=learner.context_bonus_scale,
                                 tolerance=min(1e-8,threshold/20),max_iterations=500)
            solver.warm=initial.copy()
            try:
                p,iterations=solver.solve(costs)
                if not np.isfinite(p).all() or p.min()<0 or abs(p.sum()-1)>1e-12 or np.any(learner._reachable_M@p<=0):
                    raise CandidateFailure('Refinement violated simplex/context support')
                certificate=float(learner._certificate(p,costs))
                if not np.isfinite(certificate) or certificate>threshold:
                    raise CandidateFailure(f'Refinement gap {certificate} exceeds {threshold}')
            except (CandidateFailure,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                attempts.append({'initial':label,'accepted':False,'reason':str(exc)})
                continue
            event={'round':int(t),'initial':label,'extra_newton_iterations':int(iterations),
                   'global_normalized_gap':certificate,'accepted_policy_sha256':hashlib.sha256(p.tobytes()).hexdigest(),
                   'minimum_action_probability':float(p.min()),
                   'minimum_reachable_context_probability':float((learner._reachable_M@p).min()),
                   'original_error':str(original_error),'earlier_attempts':attempts,
                   'failed_normalized_costs':costs.tolist(),'cumulative_context_estimates':learner.G.tolist()}
            self.rescues.append(event)
            learner._context_newton.warm=p.copy()
            learner._fast_counts['post_failure_context_refinement']=learner._fast_counts.get('post_failure_context_refinement',0)+1
            learner._fast_max_gap=max(learner._fast_max_gap,certificate)
            learner._last={'solver':'post_failure_context_working_set_refinement','solver_status':'independently_certified',
                'normalized_global_gap_bound':certificate,'maximum_certified_normalized_gap':learner._fast_max_gap,
                'unnormalized_global_gap_bound':certificate*learner._coefficient(t),
                'solver_iterations':int(iterations),'fast_solver_counts':dict(learner._fast_counts),
                'fast_solver_fallbacks':dict(learner._fast_failures),
                'minimum_action_probability':float(p.min()),
                'minimum_reachable_context_probability':float((learner._reachable_M@p).min()),
                'context_bonus_scale':learner.context_bonus_scale,**bonuses(learner.M,p)}
            return p
        raise NumericalSolverError('Post-failure Context refinement failed without changing tolerance: '+json.dumps(attempts)) from original_error

    def update(self,*args,**kwargs):return self.learner.update(*args,**kwargs)

    def diagnostics(self):
        return {**self.learner.diagnostics(),
                'context_rescue_code_sha256':rescue_source_sha256(),
                'context_rescue_count':len(self.rescues),'context_rescue_events':self.rescues,
                'accepted_prefix_policy_sha256':self.prefix_digest.hexdigest() if self.expected_prefix_rounds is not None else None,
                'accepted_prefix_policy_verified':self.prefix_verified}


def wrap_context_learner(learner,expected_prefix_rounds=None,expected_prefix_sha256=None):
    """Preserve non-Context learners; refine Context only after original failure."""
    if getattr(learner,'geometry',None)!='context':return learner
    if getattr(learner,'lambda_alpha',0.)!=0:raise ValueError('This rescue supports raw Context without alpha only')
    return ContextFailureRefinement(learner,expected_prefix_rounds,expected_prefix_sha256)

