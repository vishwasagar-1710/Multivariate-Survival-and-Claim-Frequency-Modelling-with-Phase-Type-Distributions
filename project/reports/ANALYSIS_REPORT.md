# Multivariate Survival and Claim Frequency Modelling with Phase-Type–Inspired Features
### Analysis Report — French Motor Third-Party Liability (freMTPL2) Portfolio

**Dataset:** 678,007 motor TPL policies, 24,938 with at least one observed claim
(a joined `freMTPL2freq`/`freMTPL2sev`-style extract, French motor insurance,
via CASdatasets / Charpentier, *Computational Actuarial Science with R*).

---

## 1. Problem Framing

Traditional pricing pipelines model claim **frequency** (a GLM on claim
counts) and claim **severity** (a GLM on claim amounts, conditional on a
claim occurring) as two independent processes, then multiply the fitted
means to get a pure premium. This project instead treats
`(log-frequency, log-severity)` as a single **bivariate** random vector,
tests whether frequency and severity are actually independent, and uses
the resulting joint structure — plus a phase-type/MAP-inspired feature
set — to build a classifier that flags high-risk policies *before*
their next renewal, i.e. using only rating factors available at
underwriting time.

---

## 2. Bivariate Normal (MVN) Model of (log-Frequency, log-Severity)

Fit directly to the 24,938 claimant policies:

| Parameter | Value |
|---|---|
| Mean, log-Frequency | 0.598 |
| Mean, log-Severity | 6.855 (≈ €948 median-scale severity) |
| Var, log-Frequency | 0.609 |
| Var, log-Severity | 1.244 |
| Cov(log-Freq, log-Sev) | 0.092 |
| **Correlation ρ** | **+0.106** |

**Interpretation.** The correlation is positive, small, but non-trivial
(≈0.11) — policies that file *more frequently* also tend, on average, to
report *modestly larger* claims, which is the opposite direction of the
naive intuition in the problem statement (that frequent claimants file
smaller claims) but is consistent with what's observed in French motor
TPL: repeat claimants tend to be higher-mileage, higher-exposure risks
whose incidents are also somewhat more severe. This is exactly the kind
of dependence a frequency-only or severity-only GLM cannot see, and it
means multiplying independently fitted frequency and severity means
will *understate* the joint tail risk of the worst policies.

See `figures/mvn_contours.png` for the fitted density over the observed
claim scatter.

### 2.1 Conditional Distribution of Severity | Frequency

Using the standard MVN conditioning identity
`S | F=f ~ N(μ_S + (σ_FS/σ_FF)(f-μ_F), σ_SS - σ_FS²/σ_FF)`:

| Scenario | Conditional mean severity (EUR) |
|---|---|
| At the *average* frequency | ≈ €1,755 |
| At frequency **2 standard deviations above average** | ≈ €2,220 (+26.5%) |

A policy whose observed frequency sits two standard deviations above the
claimant mean is expected to see severities over a quarter higher than
average — a concrete, quantified frequency→severity spillover that a
univariate severity model would completely miss.

## 3. Mahalanobis Distance / χ² Outlier Detection

Squared Mahalanobis distance of each claimant from the fitted MVN mean
was compared to a χ²(df=2) reference distribution (99% threshold = 9.21).

- **889 of 24,938 claimant policies (3.56%)** exceed the 99% χ² threshold.
- Under a correctly-specified bivariate normal we'd expect ~1% (≈249
  policies) to exceed this threshold by chance alone — observing 3.56%
  indicates the true claim distribution has **heavier tails** than a
  pure MVN (as expected for insurance losses), i.e. there is a
  meaningful population of statistically anomalous claims worth
  routing to manual/fraud review. See `figures/mahalanobis_outliers.png`.
- This diagnostic is exposed live via the API's `/claim-anomaly` endpoint
  so a mid-term claim can be screened the moment its severity is known.

## 4. MANOVA — Does Claim Behaviour Differ Across Policyholder Segments?

MANOVA (Pillai's trace) was run on the joint `(log-Frequency, log-Severity)`
vector against three segmentation variables:

| Segment | Pillai's trace | F | df | p-value | Conclusion |
|---|---|---|---|---|---|
| **Region** (22 levels) | 0.0162 | 9.69 | (42, 49832) | 5.5×10⁻⁶¹ | **Highly significant** |
| **Area** (6 levels) | 0.0043 | 10.79 | (10, 49864) | 1.5×10⁻¹⁸ | **Highly significant** |
| **VehGas** (Diesel/Regular) | <0.001 | 1.10 | (2, 24935) | 0.334 | Not significant |

**Interpretation.** Geography (Region, Area — a proxy for urban density,
road conditions, local repair costs) has a statistically significant
effect on the *joint* claim vector, confirming that territory should
remain a rating factor and is a legitimate axis for risk segmentation.
Fuel type alone does not meaningfully shift the bivariate claim profile
once other factors are held fixed at the raw/unconditional level. (The
small Pillai's trace values alongside significant p-values are typical
at n≈25,000 — statistical significance here reflects sample size more
than a large practical effect size; segment-level differences are real
but modest in magnitude.)

## 5. Phase-Type / MAP-Inspired Feature Engineering

True phase-type (PH) / Markovian Arrival Process (MAP) estimation
requires transition-level (per-claim-event) timing data not present in
this cross-sectional extract. In its place, four engineered covariates
proxy the *latent risk state* a PH/MAP model would otherwise infer from
inter-claim transition intensities:

| Feature | Proxy for |
|---|---|
| `RiskLoadIndex` | Accumulated bonus-malus "penalty state" — a discretized stand-in for a policy's position along a Markov chain of risk states |
| `VehicleRiskIndex` | Instantaneous hazard contribution of vehicle power decayed by age (power/age interaction) |
| `DriverExperienceProxy` | Position along a driver's experience/aging "phase" |
| `UrbanicityIndex` | Environmental hazard-rate multiplier (traffic density) |

These, plus the raw rating factors, feed both the bivariate MVN
diagnostics and the downstream classifier.

## 6. Risk Classifier: QDA vs. Gradient Boosting

Every policy (n=678,007) was labelled **Low / Medium / High** risk from
its realized pure premium (`ClaimTotal / Exposure`): Low = zero claims
(~96.3%), Medium/High = a median split of the positive-pure-premium
policies (~2.2% / 1.5%). A 75/25 stratified train/test split was used.

**QDA** (course-required MVN-based technique — fits a class-conditional
multivariate normal per risk tier over the engineered feature space,
with balanced priors to counteract the extreme class imbalance):

- Macro-average AUC (one-vs-rest): **0.686**
- Recall: High 44.8%, Medium 75.7%, Low 38.2% — QDA trades precision for
  sensitivity on the minority classes by design (balanced priors).

**Gradient Boosting** (flexible ML benchmark, `HistGradientBoostingClassifier`):

- Macro-average AUC (one-vs-rest): **0.745**
- At default decision threshold, GBM (like most classifiers on a ~96%
  majority class) collapses toward predicting "Low" for nearly every
  policy — useful ranking power (AUC), but a poor *hard classifier*
  without probability-threshold tuning.

**Why both AUCs sit around 0.7, not 0.95:** this is expected and
realistic. Whether an *individual* policy generates a claim in a given
year is dominated by chance; published frequency GLMs on this exact
dataset typically report Gini/AUC-equivalent figures in a similar range.
The practical takeaway (and the reason the API exposes continuous
probabilities rather than forcing a single hard label) is that
rating factors give real, but bounded, lift for *ranking* policies by
risk — which is exactly what a pre-renewal underwriting triage tool
needs, rather than a guarantee about any one policy.

Full confusion matrices and classification reports:
`reports/model_evaluation.json`.

## 7. Deployed Artifact

A FastAPI service (`api/main.py`) wraps both models:

- `POST /score` — rating factors in, `risk_tier_qda`, `risk_tier_gbm`,
  full class probabilities, and a blended `combined_high_risk_score`
  (mean predicted P(Medium)+P(High) across both models) with a
  `flag_for_review` boolean.
- `POST /claim-anomaly` — observed (frequency, severity) in, Mahalanobis
  distance + χ² outlier flag out, for mid-term claim triage.
- `GET /schema` — the exact feature contract, useful for building a
  front-end or bulk-scoring job against the API.

See `README.md` for how to run it locally, in Docker, or in a GitHub
Codespace, and `VIDEO_SCRIPT.md` for a walkthrough script.

## 8. Limitations & Honest Caveats

- **Class imbalance dominates.** ~96% no-claim base rate means any
  hard-classification accuracy number is misleading; AUC and the
  probability outputs are the metrics that matter here.
- **MVN is an approximation.** The 3.56%-vs-1% Mahalanobis exceedance
  rate confirms real claim severities are heavier-tailed than Gaussian;
  a log-normal/MVN approximation is a reasonable, tractable first
  layer, not a terminal model — a Gamma/Pareto-tailed copula would fit
  the extreme tail better in a follow-up iteration.
- **No true phase-type/MAP estimation.** Section 5's features are
  theory-inspired proxies, not a fitted MAP; a genuine MAP/PH fit needs
  claim-event timestamps this cross-sectional extract doesn't provide.
- **Labels are historical, not causal.** The Low/Medium/High target is
  built from the same policies' own realized experience, so the
  classifier is validated on held-out *policies*, not a held-out *time
  period* — a production deployment should re-validate on a genuinely
  future policy year before being used to change underwriting decisions.
