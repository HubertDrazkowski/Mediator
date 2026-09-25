# Algorithm references

These are the baseline references used by the accompanying paper and implementations. The parameter table, native Tsallis objectives, numerical conventions and explicitly labelled experimental extensions are specified in [ALGORITHMS.md](ALGORITHMS.md).

| Method | Source used by the implementation |
|---|---|
| Tsallis-INF | Zimmert and Seldin (2021), *Tsallis-INF: An Optimal Algorithm for Stochastic and Adversarial Bandits*. The package's adopted main coefficient and RV cutoff are specified in [ALGORITHMS.md](ALGORITHMS.md). |
| EXP4MF (IW) | Eldowa et al. (2024), Algorithm 1 and the BOBW schedule of Theorem 3. [Source, Section 4](https://arxiv.org/html/2402.10282v1#S4). |
| PE | Liu, Attias and Roy (2024), Algorithm 2, Phased Elimination via a linear-bandit reduction. [Source algorithm](https://arxiv.org/html/2407.00950v1#alg2). |
| CUCB | Lu et al. (2020), Algorithm 1, C-UCB; the default fixed confidence uses the horizon. [Source, page 4](https://proceedings.mlr.press/v124/lu20a/lu20a.pdf#page=4). |
| CTS | Lu et al. (2020), Algorithm 2, C-TS with independent Beta priors for mediator reward means. [Source, page 5](https://proceedings.mlr.press/v124/lu20a/lu20a.pdf#page=5). |
| CUCB2 | Nair, Patil and Sinha (2021), Section 5 / Algorithm 3, C-UCB-2. [Source, page 10](https://arxiv.org/pdf/2012.07058#page=10). |

The first reference concerns the action-only Tsallis-INF baseline. The mediator objective and minimum-regularizer CTsallis family belong to the accompanying paper; they are not attributed to that baseline reference. Their implemented definitions are explicit in [ALGORITHMS.md](ALGORITHMS.md) and `replication/algorithms`. Stable identifiers retain earlier mathematical labels, while figures use CTsallis-Mediator and CTsallis-INF.

EXP4MF (RW) is the package's reduced-variance estimator extension of EXP4MF. Its shared policy/rate schedule is based on Eldowa et al. (2024); the modified estimator is not asserted to be the source paper's algorithm or to inherit its theorem. Likewise, the online estimated-mechanism variants are the explicit plug-in extensions documented in [ALGORITHMS.md](ALGORITHMS.md), rather than additional algorithms attributed to the baseline papers.

The numerical runtime depends on NumPy, SciPy, CVXPY and Clarabel. The package retains explicit objective/design certificates and reports a numerical failure when its acceptance conditions cannot be met.

## Public empirical data

Project STAR: the public `STAR` data distributed with AER, documenting
Tennessee's randomized class-size study (1985--1989). The source file,
checksum, base-R export, variable meanings, cohort definition, and empirical
benchmark limitations are specified in [STAR.md](STAR.md). Data are not
redistributed in this code package. Public data source:
https://raw.githubusercontent.com/cran/AER/master/data/STAR.rda
