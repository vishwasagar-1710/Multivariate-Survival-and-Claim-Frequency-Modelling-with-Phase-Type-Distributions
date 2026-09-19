# Video Walkthrough Script (target: 4–6 minute LinkedIn post)

Talking points, not a verbatim transcript — deliver in your own voice.
Suggested screen order noted in **[brackets]**.

---

### 1. Hook (15–20 sec) **[Face to camera / title slide]**
- "Actuaries traditionally price frequency and severity separately.
  I wanted to test whether that independence assumption actually
  holds — and build something that uses the answer."
- State the dataset: 678K French motor third-party-liability policies.

### 2. The core statistical question (45–60 sec) **[figures/mvn_contours.png]**
- Explain you modeled `(log claim frequency, log claim severity)` as
  a joint bivariate normal instead of two separate models.
- Show the contour plot over the claimant scatter.
- Give the headline number: **correlation ρ ≈ +0.11** — frequent
  claimants trend toward *slightly larger* claims, not smaller ones.
  Call out that this is genuinely counter to the "high-frequency,
  low-severity" folk wisdom, and explain why (repeat claimants often
  correlate with higher exposure/mileage risk profiles).
- One line on the conditional-distribution result: at 2 standard
  deviations above average frequency, expected severity is ~26%
  higher — a spillover a univariate model can't see.

### 3. Are segments actually different? MANOVA (30–45 sec) **[reports table or a simple bar of p-values]**
- Explain MANOVA tests whether the *whole* claim vector shifts across
  a segment, not just frequency or just severity in isolation.
- Result: Region and Area are highly significant (p ≈ 10⁻¹⁸ to 10⁻⁶¹);
  fuel type (Diesel vs Regular) is not (p = 0.33).
- Takeaway: geography earns its place as a rating factor; fuel type,
  on its own, doesn't move the joint claim profile much.

### 4. Catching anomalous claims (30–45 sec) **[figures/mahalanobis_outliers.png]**
- Explain Mahalanobis distance as "how many standard deviations away
  from typical claim behaviour, accounting for the frequency-severity
  correlation."
- Show the histogram vs. the chi-square(df=2) reference curve.
- Result: 3.56% of claimants exceed the 99% threshold — more than
  three times the ~1% you'd expect under a perfect Gaussian fit,
  meaning real claims have heavier tails than a normal approximation
  — exactly the population you'd want flagged for manual/fraud review.
- Mention this is a live API endpoint (`/claim-anomaly`), not just a
  static chart.

### 5. From statistics to a deployed classifier (45–60 sec) **[terminal / Swagger UI at /docs]**
- Explain the risk-scoring goal: flag Low/Medium/High risk *before*
  renewal, from rating factors alone — no claims from the new term
  are used.
- Two models: QDA (the multivariate-normal classifier straight out of
  the course theory) and a Gradient Boosting benchmark.
- Be honest about the numbers: macro AUC ~0.69 (QDA) and ~0.75 (GBM) —
  explain *why* that's actually a believable, "real insurance" result
  (claims are dominated by chance at the individual level; the value
  is in ranking policies, not guaranteeing outcomes).
- Live demo: hit `POST /score` with a rough, low-mileage, older-driver
  profile vs. a young-driver/high-power/urban profile; show the flip
  from a "Low" to a "High" flag and read out the combined score.

### 6. Deployment (15–20 sec) **[GitHub Codespace tab]**
- Show the Codespace spinning up, devcontainer auto-installing and
  training, `bash run_api.sh`, and the forwarded port opening
  `/docs`.
- One line: "Fully reproducible — clone it, open a Codespace, and the
  whole pipeline retrains itself."

### 7. Close (10–15 sec) **[Face to camera]**
- Recap in one sentence: joint modeling changes both the risk story
  (frequency and severity aren't independent) and the tooling
  (a single API endpoint that scores and flags a policy in one call).
- Call to action: link to the repo in the post; invite comments on
  the MVN-vs-heavier-tail tradeoff or alternative segment variables to
  test.

---

## B-roll / screenshot checklist
- [ ] `figures/mvn_contours.png`
- [ ] `figures/mahalanobis_outliers.png`
- [ ] `reports/ANALYSIS_REPORT.md` scrolled to the MANOVA table
- [ ] Swagger UI (`/docs`) with `/score` expanded
- [ ] Terminal showing `bash run_api.sh` starting cleanly
- [ ] GitHub Codespaces "Create codespace on main" button + first boot

## Suggested LinkedIn post caption (short form)

> Claim frequency and severity aren't independent — at least not in
> this French motor TPL portfolio. I modeled them jointly as a
> bivariate normal, ran a MANOVA to check which rating factors
> actually shift the *joint* claim profile, used Mahalanobis distance
> to flag statistically anomalous claims, and shipped a QDA +
> Gradient Boosting risk-scoring API you can spin up in one click via
> GitHub Codespaces. Full write-up and repo in the comments. #actuarial
> #machinelearning #insurance #datascience
