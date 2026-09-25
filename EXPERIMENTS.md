# Experiment definitions

All matrices have shape **Z × A** and columns sum to one. Actions and mediators
are indexed from zero; rounds and phase starts are indexed from one. In the synthetic experiments, for a chosen action A_t, the simulator draws
Z_t ~ M[:,A_t], followed by a Bernoulli loss with mean g_t(Z_t). STAR instead
samples entire empirical records within the chosen action, preserving the joint
mediator/loss distribution and non-binary bounded losses. Learners receive only their supplied mechanism and
the selected action/observed mediator/loss. Loss means, gaps, optimal actions
and complexity diagnostics are evaluator-only information.

## 1. E1: stationary and two common-baseline switching regimes

For (A,Z) in {(80,30),(30,80),(60,60)},

```
M[z,a] = (1-epsilon) * 1[z == a % Z] + epsilon/Z
epsilon = 0.3
```

Stationary means are (0.45,0.55,...,0.55). Switching starts with
(0,0.1,...,0.1), then alternates with (0.9,1,...,1). The segment-length state
starts at 50; each segment uses its ceiling, and the state is multiplied by
1.6. The final segment is truncated. Thus k=0,1,... indexes successive
segments. The implementation uses iterative multiplication, matching the
original floating-point schedule exactly. At T=30,000 its starts are
1,51,131,259,464,792,1317,2156,3499,5647,9083,14581,23378.

The mechanism, optimal-action set and action gaps remain fixed throughout;
learners are never reset. The positive action gap is 0.07. A80/Z30 has three
identical optimal-action columns (actions 0,30,60); A30/Z80 has 50 mediators
observed only through the shared background. This matters when interpreting
the cardinalities or unique-optimum assumptions. Every mediator is observed
with positive probability under optimal play, so the opening-support
attainable mediator complexity is zero. The raw mediator-loss gap 0.1 is a
different quantity.

The random piecewise regime uses g_s=clip(g0+beta_s,0,1), g0=(.45,.55,...),
with independent beta_s~Uniform(-.45,.45) on the same segments. Its random offsets
are frozen per trajectory seed and shared across methods and dimension pairs;
clipping is inactive at the defaults. It changes loss baselines and Bernoulli
variances but not action gaps. No method restarts at segment boundaries.

Each box contains 30 trajectory seeds. Stationary and deterministic-switching
boxes vary observation/learner randomness on one fixed schedule. Random-piecewise
boxes also vary the schedule, paired across methods. The focus figure shows
A80/Z30 stationary and deterministic switching; the full figure has three regimes
by three dimension pairs.

## 2. Random action-mediator graphs: stationary and deterministic switching

The stationary sample combines the original 50 graphs (seed 2026091802) and
an independent 20-graph extension (seed 2026092301), with one loss vector and
one trajectory per method per graph (trajectory seed 8201). The two batches
are generated separately: the balanced generator is not prefix-stable in its
requested sample size. The settings are assigned
by independently randomized balanced cycles: each level of each factor has a
count differing by at most one. Dimension-pair levels are the nine ordered
pairs from {10,30,80}². This balances marginal factors rather than all joint
combinations.

Randomly assign actions and mediators to two balanced groups. Each edge is
sampled independently with probability d(1+r) within groups and d(1-r) across
groups, for d in {0.05,0.2,0.4}, r in {0,0.5,1}. Repair each isolated vertex
with one uniformly chosen allowable edge; at r=1 repairs stay within groups.
Record repairs because actual density can exceed the pre-repair target.
On each action's resulting support, sample independent Dirichlet(kappa,...,
kappa) weights, kappa in {0.3,1,10}.

Independently draw x_z iid Uniform[-1,1] and set g_z=0.5+lambda*x_z, with lambda
in {0.03,0.1,0.3}. Derive action means mu=M.T@g and gaps Delta_a=mu_a-min(mu).
No action gaps are independently imposed. The graph and means stay fixed
throughout a trajectory.

The diagnostic S_A sums 1/Delta_a over suboptimal actions. Let U denote the
reachable mediators outside the support of the chosen optimal action. The
jointly attainable mediator complexity is computed as

```
S_Z^ach = min sum_{z in U} 1/gamma_z
          subject to gamma_z > 0,
          sum_{z in U} M[z,a]*gamma_z <= Delta_a  for every suboptimal action a.
```

This joint optimization differs from optimizing each margin separately.
Primal/dual bounds, ties and unresolved diagnostics are saved. These diagnostics are exported but do not split the main random-graph figure.
All70 environments remain in its stationary aggregate, including tied optima.
CUCB2 N/A cases remain in the CSV and figure.

Regret is normalized separately on each instance by the expected regret of a
uniform action policy, T*(mean(mu)-min(mu)). Variation in the random-graph boxes
is across environments; with one trajectory per instance it does not separate
environment variation from trajectory noise.

The deterministic-switching sample selects 30 of these 70 environments using the
smallest SHA256 hashes of `2026092401|scenario_id`, independently of performance.
For each selected stationary mean vector g0, set gL=g0-min(g0) and
gH=1-(max(g0)-g0). Alternate low/high on the E1 geometric schedule, starting low.
There is no clipping or resampling of graphs. M, all action gaps, optimal actions,
and the uniform-policy normalizer are unchanged. Paired environment uniforms use
the stationary scenario ID in both regimes. Both the 70-versus-30 overview and the
matched30-versus-30 comparison are exported. Dashed red horizontal lines show
CTsallis-INF's sample median across each whole panel.

## 3. Online estimation of the mechanism

Use the same six E1 environments, 30 trajectory seeds, and three conditions:
true M; an online estimate initialized with 500 sampled pairs; an online
estimate initialized with 2,000 sampled pairs. An initial pair is generated by
sampling an action uniformly and then a mediator conditionally from M[:,a].
It is **not** a uniform draw from the Cartesian product of action/mediator
labels. There are no initial loss samples.

Within a seed and dimension pair, the first 500 pairs are a prefix of the
2,000 pairs; all methods and both loss regimes share this initial dataset.
With total prior strength one per action, the matrix used on round t is

```
Mhat_t[z,a] = (N_initial[z,a] + N_online_before_t[z,a] + 1/Z)
             / (N_initial[a] + N_online_before_t[a] + 1).
```

The learner selects actions adaptively after initialization. There is no extra
uniform online exploration. The round-t estimator and loss update use Mhat_t;
only then is the new pair counted. Historical estimated action losses are
retained without remapping them through newer matrices. Current regularization
uses the current fitted matrix. PE adopts the current estimate at phase
boundaries and keeps features/design fixed within a phase. Learning histories
are never reset. True M generates feedback and evaluation regret only.

These are specified plug-in extensions: known-M regret guarantees are not
claimed for estimated M. Tsallis-INF ignores M; its repeated condition rows
within a seed are therefore duplicate comparisons, not independent samples.

## Regret, randomness, and failures

Cumulative pseudo-regret is sum_t [mu_t(A_t)-min_a mu_t(a)], based on the
actually selected actions and true means, not sampled losses. The runner also
records sum_t [p_t.T@mu_t-min(mu_t)] as policy pseudo-regret. These two quantities
have the same expectation but different trajectory variability. Figures use
the former. For the baseline switches the optimal-action set stays fixed, so
the instantaneous and fixed-best-action comparators agree.

For each environment/trajectory seed, the same array of action, mediator and
loss uniforms is supplied to all methods and mechanism conditions. Learner
internal randomness uses a separate deterministic stream. Changing method
ordering, worker count or run scheduling does not change these streams.
Algorithms may choose different actions under the shared uniforms.

Numerical failures and structural incompatibilities remain in the planned
population. Successful earlier checkpoints remain available. No result is
imputed and no environment is replaced based on performance. Source hashes and
full parameter specifications are frozen before learner execution.

## 4. Complete-graph transpose adaptation

For n in {8,12,16}, form the vertex-edge incidence matrix of the complete
undirected graph. With vertex actions, M[e,v]=1[v belongs to e]/(n-1). With edge
actions, M[v,e]=1[v belongs to e]/2. Thus cardinalities are (n,choose(n,2)) and
(choose(n,2),n). Each column is uniform on an equal-size support; the entropy
calibration subtracts the same constant across actions within a setting.

Choose one optimal vertex/edge uniformly using the declared environment seed,
then hold it fixed across all 30 trajectory seeds10401--10430. Let b=.35, D=.02.
For vertex actions, g(e)=b+D*(n-1)/(n-2) if the optimal vertex is not in e,
and b otherwise. For edge actions, g(v)=b+2D if v is not an endpoint of the
optimal edge, and b otherwise. Both orientations have one optimal action and
minimum positive action gap D. Losses are Bernoulli and stationary.

The action/attainable-mediator complexity ratio is2*(n-1)/(n-2)^2 for vertex
actions and(n+5)/2 for edge actions. Analytic feasible margins and matching dual
weights are saved as evaluator diagnostics. These are not learner inputs.
Increasing n strengthens the contrast without changing the minimum action gap.
Only CTsallis-Action, CTsallis-Mediator, and CTsallis-INF run, at T=50,000 with
unchanged coefficients. Dashed red segments extend INF's sample median leftward.

## 5. Project STAR

Use the public AER STAR dataset, retaining4,298 records with known kindergarten
class type and observed first-grade reading and mathematics scores. Actions are
regular, regular with an aide, and small kindergarten classes. The mediator is
the quartile of the average kindergarten reading/math score, with an additional
missing-score state. Quartile cutpoints435,457.5,482 are defined using 5,786 rows
with known action and both kindergarten scores, before requiring later outcomes.

Loss is one minus the midrank empirical CDF of the average first-grade reading/
math score relative to the original 3,999 complete cases. The reference remains
fixed when299 missing-mediator records are added. See STAR.md for exact source
and transformations. No outcomes are imputed.

Each round samples one entire record uniformly with replacement within the
selected treatment arm. The mediator and bounded non-binary loss are revealed
together. The true mechanism is the empirical frequency of mediator states in
each arm and is supplied to learners. True raw arm mean losses determine
pseudo-regret; pooling losses by mediator for simulation would wrongly impose
perfect mediation. The original row order is retained for exact RNG replay.

The main experiment uses five paired seeds27001--27005, all ten standard
comparison methods, and T=20,000. The figure uses a logarithmic regret axis.
This is a fixed-cohort empirical robustness experiment, not a clinical or
educational policy recommendation and not identification of a direct effect.
