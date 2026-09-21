# Multivariate Survival and Claim Frequency Modelling with Phase-Type Distributions

**Technical report** — all figures below are pulled directly from
`outputs/results/results.json`, produced by `scripts/run_pipeline.py`
against the real **French Motor Third-Party Liability Claims** dataset
(freMTPL2freq.csv + freMTPL2sev.csv, 678,013 policies, Charpentier /
CASdatasets). Nothing in this report is hand-computed.

---

## 1. Problem statement

Standard actuarial pricing fits claim frequency (a Poisson/Negative-Binomial
GLM) and claim severity (a Gamma/log-normal GLM) as two independent models,
then multiplies them for the pure premium. This project tests whether that
independence assumption actually holds, quantifies how far real claim
behaviour departs from a simple bivariate-normal approximation, and turns
the findings into a pre-renewal risk-scoring tool.

## 2. Data

- **Source**: real freMTPL2 dataset, 678,013 policy-years.
- **Cleaning** (see `src/data_processing.py::load_real_kaggle_data`,
  matching the standard treatment used in the actuarial ML literature,
  e.g. scikit-learn's own Tweedie-regression tutorial on this same data):
  Exposure clipped to 1.0 (max raw value was 2.01, exceeding a policy-year);
  ClaimNb clipped to 4 (a handful of rows report up to 16, treated as data
  errors); ClaimNb reset to 0 for 9,116 rows where the matched ClaimAmount
  was 0 despite ClaimNb >= 1 (a known data-entry artifact); ClaimAmount
  clipped at 200,000 to prevent a handful of multi-million-euro claims from
  dominating variance estimates.
- **Claim rate after cleaning**: 3.68% of policies have at least one claim.
- **Claimant subsample used for the MVN/MANOVA/QDA stages**: 24,944 policies.

A synthetic fallback generator (`data/generate_data.py`) is also included,
built around a known Gaussian-copula dependence structure between
frequency and severity. It exists so the pipeline is runnable and testable
without the real data, and so the statistical machinery could be validated
against a *known* ground-truth correlation before trusting it on the real
data (see `tests/test_pipeline.py`, 25/25 passing). All headline results
in this report use the real data, not the synthetic set.

## 3. Bivariate MVN fit: (log-frequency, log-severity)

For the 24,944 claimant policies:

```
mu    = [0.599, 6.855]           (log-frequency, log-severity)
Sigma = [[0.611, 0.093],
         [0.093, 1.238]]
rho(frequency, severity) = +0.106
```

**Finding**: the correlation is small but reliably **positive**, not
negative. This is worth stating plainly because it *contradicts* the
"high-frequency claimants file smaller claims" stylized fact quoted in
many textbook treatments of this topic. On this particular book of
business, policies that claim more often also trend (mildly) toward
larger average claims, not smaller ones. A plausible reading: BonusMalus
and Exposure — the two dominant drivers of both dimensions (see Section
6) — push frequency and severity in the same direction simultaneously
(a policy with high exposure has more opportunity for both frequent *and*
large claims), rather than there being a direct causal claimant-behaviour
link between how often someone claims and how much each claim costs. This
is exactly the kind of finding an independence-assumption pricing model
would miss entirely, and it is the central empirical justification for
this project's joint-modelling approach — whichever direction the
dependence runs, ignoring it means ignoring real structure in the data.

### Conditional distribution of severity given frequency

Using the standard MVN conditioning formula
`X2 | X1=x1 ~ N(mu2 + (Sigma21/Sigma11)(x1-mu1), Sigma22 - Sigma21^2/Sigma11)`:

| Frequency percentile | Conditional median severity |
|---|---|
| 10th | EUR 866 |
| 50th (median) | EUR 907 |
| 90th | EUR 1,106 |

Consistent with rho > 0: conditional expected severity rises modestly
with frequency across this range.

## 4. Mahalanobis distance / chi-square outlier test

Under H0 (bivariate normality), `D^2 = (x-mu)'Sigma^-1(x-mu) ~ chi2(2)`.

- Critical value at alpha=0.01: **D^2 = 9.21**
- **895 of 24,944 claimants (3.59%)** exceed it — well above the 1% that
  true bivariate normality would predict.
- KS test of the empirical D^2 distribution against chi2(2): **p < 0.001**,
  rejecting exact bivariate normality.

**Interpretation**: real claims data is heavier-tailed than a bivariate
normal — expected, since insurance severities are famously right-skewed
even after a log transform. The MVN fit is a useful *first-order*
approximation of the dependence structure (and is exactly what the
project brief calls for: an "MVN-approximated copula"), but the
Mahalanobis/chi-square diagnostic is itself evidence that a full copula
model with a heavier-tailed marginal (e.g. a t-copula, or a genuine
PH/MAP-based severity marginal) would fit better. This is flagged as
future work in Section 8, not glossed over.

## 5. One-way MANOVA: claim behaviour across policyholder segments

Segments: {Young, MidAge, Senior} x {Urban, Rural} (driver age and
population density of residence), tested jointly on
(LogFrequency, LogAvgSeverity).

```
Wilks' Lambda  = 0.9718
Pillai's Trace = 0.0282
F(10, 49874)   = 71.84
p-value        < 1e-10  (effectively 0)
```

**Segments differ significantly** in joint claim behaviour. Group means:

| Segment | Log-Frequency | Log-Severity |
|---|---|---|
| Young-Urban | 0.851 | 6.896 |
| Young-Rural | 0.850 | 6.860 |
| MidAge-Urban | 0.639 | 6.846 |
| MidAge-Rural | 0.552 | 6.835 |
| Senior-Urban | 0.492 | 6.925 |
| Senior-Rural | 0.365 | 6.976 |

Young drivers claim noticeably more often (log-frequency ~0.85 vs. ~0.36
for rural seniors) regardless of urban/rural status — age dominates the
frequency dimension. Severity is comparatively flat across segments
(6.83-6.98 on the log scale, a narrow band), with a slight tendency for
**seniors to have larger average claims** despite claiming less often —
consistent with the "fewer but costlier" pattern often observed for
older-driver accidents, and, notably, consistent with the positive
frequency-severity correlation being driven by *other* factors
(Exposure/BonusMalus) rather than age, since age moves the two dimensions
in *opposite* directions here.

## 6. QDA risk-tier classification

Labels: tertiles of `risk_score = LogFrequency + LogAvgSeverity` (a log
expected-cost proxy) among claimants -> Low / Medium / High.

```
Train accuracy: 83.9%
Test accuracy:  84.3%
```

| | Precision | Recall | F1 |
|---|---|---|---|
| High | 0.96 | 0.91 | 0.94 |
| Medium | 0.70 | 0.97 | 0.82 |
| Low | 0.95 | 0.65 | 0.77 |

QDA (rather than LDA) was used deliberately: the tiers are hypothesised to
have *different* covariance structures (the High tier, in particular, has
a visibly fatter joint spread in `outputs/plots/05_qda_decision_boundary.png`),
which only a quadratic (per-class covariance) boundary can represent. The
Medium tier has the weakest precision (0.70) — it is the tier squeezed
between two others on a continuous underlying score, so some boundary
bleed is expected and is not a modelling error.

## 7. MAP/PH-inspired feature engineering

Per rating cell (Area x VehGas x DrivAgeBucket, minimum 15 claimants),
severities are moment-matched to a 2-phase phase-type distribution:

- **SCV > 1** -> 2-phase **hyperexponential** (heavier than exponential tail)
- **SCV < 1** -> **Coxian-2 / Erlang-mixture** (more regular than exponential)
- **SCV = 1** -> single-phase **Exponential**

using the balanced-means moment-matching formulas
(`src/ph_features.py::fit_ph2_moment_match`). This is fit per rating cell,
not per individual policy, since a single policy with 1-2 claims does not
carry enough information to estimate a distribution's variance — the cell
aggregate does. The resulting descriptors (`ph_p`, `ph_rate1`, `ph_rate2`,
`ph_scv`, `cell_mean_severity`, `cell_var_severity`) are joined back onto
every policy in the cell as pre-renewal-available features for Section 8's
classifier.

## 8. Pre-renewal high-risk classifier

**Target**: `HIGH_RISK = 1` if realised `ClaimAmount` exceeds the 98th
percentile of the portfolio (EUR 1,128.55) — a "large loss" flag that
lands at a **2.0% positive rate** given this book's 3.68% claim rate
(most claims are small; the 98th-percentile cut isolates the genuinely
costly tail).

**Model**: Gradient Boosting (sklearn, 300 estimators, depth 3, balanced
sample weights for the class imbalance) on rating factors (VehPower,
VehAge, DrivAge, BonusMalus, Density, Exposure, Area, VehGas, Region,
VehBrand) plus the PH-inspired cell features from Section 7 — all
available *before* the renewal decision, with no leakage from the current
period's own claim outcome.

```
Test AUC     = 0.712
Test PR-AUC  = 0.068   (vs. 0.020 base rate -- a 3.4x lift over random)
```

At the calibrated operating threshold (flagging the riskiest 2% of the
book, matching the target prevalence): precision 0.13, recall 0.13 on the
held-out 169,504-policy test set — modest in absolute terms, which is
expected and consistent with the actuarial literature: predicting *which
specific policy* will produce a large loss from a priori rating factors
alone is a genuinely hard problem (large losses have substantial
irreducible randomness). An AUC of 0.71 represents real, useful separation
for portfolio triage (e.g., routing the top-decile-scored renewals to
manual underwriting review) even though it will never approach AUC ~0.95+
territory.

**Top predictors** (Gradient Boosting feature importance):

| Rank | Feature | Importance |
|---|---|---|
| 1 | Exposure | 0.431 |
| 2 | BonusMalus | 0.307 |
| 3 | DrivAge | 0.090 |
| 4 | VehAge | 0.054 |
| 5 | LogDensity | 0.040 |
| 6 | VehPower | 0.012 |
| 7-10 | VehBrand, VehGas, Region, ph_rate1 | < 0.011 each |

Exposure and BonusMalus alone account for **74% of total importance** —
unsurprising (longer exposure = more opportunity to claim; BonusMalus is
France's own regulator-mandated risk score, built for exactly this
purpose), but it is a useful sanity check that the model has recovered
the two variables actuaries already know matter most, rather than
latching onto noise. The PH-derived features contribute modestly
(`ph_rate1` at rank 10) — a legitimate, if secondary, signal.

## 9. Risk-scoring API

`app/main.py` (FastAPI) exposes `POST /score` and `POST /score/batch`,
returning `{high_risk_probability, high_risk_flag, risk_tier,
operating_threshold, model_version}` for a policy's pre-renewal rating
factors. Two calibrated thresholds partition the probability space into
Low / Medium / High tiers: the operating threshold (top 2% of predicted
risk, matching target prevalence) separates Low from Medium/High, and a
stricter threshold (top 1%) separates Medium from High. See README.md
"Running the API" and "Deployment" for usage and GitHub Codespaces
instructions; `app/demo_server.py` is a dependency-free stdlib server
exposing identical logic, used to verify the pipeline end-to-end in
network-restricted environments.

## 10. Limitations and future work

1. **MVN is an approximation, not the true copula.** Section 4's
   chi-square test explicitly rejects exact bivariate normality. A next
   iteration would fit a genuine copula (Gaussian copula with non-normal
   marginals, or a t-copula for tail dependence) on top of Poisson/PH
   marginals, rather than transforming to logs and assuming normality.
2. **PH features are cell-level, not policy-level**, because individual
   policies rarely have enough claims to estimate a variance. A richer
   MAP (Markovian Arrival Process) treatment jointly modelling the
   *arrival process* of claims (not just their size) was out of scope
   here but is a natural extension given the project's MAP/PH framing.
3. **The classifier has no monotonicity constraints.** A production
   pricing/underwriting model would typically constrain risk to be
   non-decreasing in BonusMalus, for auditability and regulatory comfort;
   this prototype does not enforce that.
4. **Severity is clipped at EUR 200,000.** This is standard practice in
   the literature for this dataset (avoids a handful of multi-million-euro
   claims dominating variance) but is a modelling choice that discards
   information about the very largest losses — exactly the losses a
   reinsurance-layer analysis would care most about.
5. **The positive frequency-severity correlation is book-of-business
   specific.** It should not be read as a general claim that
   "high-frequency claimants file larger claims" — Section 3 discusses
   why it more likely reflects shared drivers (Exposure, BonusMalus)
   than a direct behavioural link, and the sign or magnitude could differ
   on another portfolio.
