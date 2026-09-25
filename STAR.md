# Project STAR: stationary empirical benchmark

The public source is the `STAR` dataset distributed with the R package AER:
[source RData](https://raw.githubusercontent.com/cran/AER/master/data/STAR.rda),
[variable documentation](https://cran.r-universe.dev/AER/doc/manual.html#STAR).
Project STAR was Tennessee's randomized class-size study (1985–1989).
The benchmark uses recorded classroom exposure, not a separately observed
randomized-assignment instrument.

## Get the data explicitly

No dataset is included in this repository. Install base R if `Rscript` is not
available; no R packages need to be installed. From the repository root run:

```bash
python scripts/fetch_star.py
python run_experiments.py --config configs/star.json --action prepare
```

The fetch script downloads the public RData source, verifies its SHA-256,
exports four columns to `data/STAR.csv`, and creates
`data/STAR.provenance.json`. It does not run any learners. The temporary RData
is discarded. For a previously downloaded copy:

```bash
python scripts/fetch_star.py --source-rda path/to/STAR.rda
```

The experiment configuration controls `star.source_path`, a relative CSV path
inside the repository. The CSV must contain `row_id`, `action`,
`mediator_value`, and `outcome_value` in the original source ordering. Missing
values are blank. The default `star.verify_paper=true` rejects changed values
or ordering. An existing correct CSV is reused; data are never fetched
automatically by an experiment or a test.

Source SHA-256:
`9e597f8096550185acb7917e6b087a45815574b0f2a4d770222395dd85a4b7b1`.
Ordered, normalized source-row SHA-256:
`942180c2c1891b9c566ddd7c3d8633bc16e55397048d0c9fa71b975f9f301dcb`.
The latter ignores CSV formatting and line endings, while detecting changes
to row order, IDs, actions, and scores. Both definitions are in
`replication/star.py`. Source and export checksums are recorded in provenance.

## Exact construction

The source contains 11,598 students. We select 6,325 with observed kindergarten
class type (`stark`). The three actions, in this exact order, are `regular`,
`regular+aide`, and `small`. The early score is `(readk+mathk)/2`, requiring
both kindergarten reading and mathematics measurements. Its three pooled
quartile cutpoints, computed on all 5,786 observed early-score records before
filtering later outcomes, are **435, 457.5, and 482**. Ties enter the upper bin.
A missing early score is a fifth mediator state, after Q1–Q4.

The later outcome is `Y=(read1+math1)/2`; both first-grade scores must be
observed. We retain 4,298 records, including 299 with a missing early score.
Their action counts are 1,456, 1,503, and 1,339. Missing later outcomes are
excluded; they are not assigned artificial loss values.

The loss is `1 - F_mid(Y)`, where
`F_mid(y) = [#{Y_ref < y} + 0.5 #{Y_ref = y}]/3999` uses the fixed reference
cohort of 3,999 records with both early and later scores observed. This
reference is preserved when adding the missing-mediator records. Losses are
bounded, non-binary values on finite support. They are not converted into
Bernoulli outcomes by the environment.

Each round selects an action and samples one whole retained record uniformly
with replacement from that arm. Its mediator and loss are revealed together.
The resulting conditional joint law `P(Z,loss | A)` is preserved; losses are
not pooled across arms. The learner receives the exact empirical transition
matrix `M[z,a]=count(z,a)/count(a)`. Pseudo-regret uses the **raw empirical arm
means**, with `R_T = sum_t(mu[A_t] - min_a mu[a])`, rather than `M.T @ pooled_g`.
The latter is retained only as a diagnostic and does not define feedback or
the benchmark optimum.

Kindergarten achievement occurs after exposure to class size and staffing and
can plausibly transmit their effect to later achievement. Coarse achievement
categories need not capture every treatment pathway, and later classroom
exposure is not conditioned away. Thus the experiment tests performance in a
stationary empirical environment with potentially incomplete mediation; it
does not assume `loss` is conditionally independent of `A` given `Z`.
Selection on later observed outcomes and recorded exposure prevents treating
this replay as an identified causal mediation effect or a population result.

## Reproduction and interpretation

`configs/star.json` specifies all ten displayed algorithms (without EXP4MF RW),
five trajectory seeds 27001–27005, and a true horizon of 20,000 rounds. Use this
horizon when constructing learners: some baseline parameters depend on the
horizon, so taking the first 20,000 rounds of a 30,000-round run is different.
The main plot uses a common logarithmic vertical axis and displays each seed.
Monte Carlo variation is conditional on the frozen empirical cohort; it is
not an uncertainty interval for population treatment effects.

To preserve the original random streams, `star_rng_seeds(seed)` uses the first
16 SHA-256 bytes of default `json.dumps` tuple encodings, interpreted little
endian: `('raw_realdata_v1','STAR',seed)` for environment uniforms, and
`('STAR',seed,'learner')` for the independent learner RNG. A `T x 2` uniform
array is generated in advance: column 0 selects the action, column 1 selects
the row within that action in original source order. The unchanged CTS
implementation internally Bernoulli-resamples the bounded reward before its
Beta update; the empirical environment itself does not modify the loss.

The raw empirical arm loss means are approximately 0.51799, 0.52712, and
0.45995, making the small-class arm best. Every arm has positive probability
for all five mediator states. Low pseudo-regret therefore means rapid
concentration on the best empirical arm, not low observed losses. Favorable
feedback coverage in this cohort does not establish general robustness to
violations of separation.

Only this main full-mediator STAR experiment is included. No ACTG experiment
or mediator-coarsening sensitivity runs are part of this package.
