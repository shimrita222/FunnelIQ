# FunnelIQ — Findings & Recommendations

A summary of what the six analytical packages found, and what Northbound Media
should actually do about it. Full methodology, code, and evaluation tables live
in [README.md](README.md) and the individual scripts/notebooks; this document
focuses on the business answers.

## The five questions, answered

### 1. How long will a new customer stay?

**Model:** CatBoost regression on `ltv_months` (5-fold CV Mean RMSE 2.95 months,
R² 0.943; held-out test RMSE 3.09, R² 0.938).

**Finding:** the number of calls it takes to close a deal (`calls_to_closed`) is
the single strongest predictor of lifetime — and it's a *negative* relationship
(r ≈ −0.66). Customers who take longer to close tend to stay for **shorter**,
not longer.

**Recommendation:** treat a fast, low-friction close as an early positive
signal for retention, not just sales efficiency. Accounts that needed many
calls to close are the group most likely to churn sooner — flag them for
proactive retention outreach right after onboarding, rather than waiting for
churn signals to show up later.

### 2. Who is likely to buy more?

**Model:** CatBoost classification on `upsell`, without `ltv_months` as a
feature (removed after a project-wide leakage audit — see below). F1 0.758,
Recall 0.860, vs. a simple business-rule baseline that manages only F1 ≈ 0.3–0.4.

**Finding:** no single feature dominates — `purchased` (~35% importance) and
the rest of the early-funnel signals together drive the prediction. The model
substantially outperforms a manual threshold rule, mainly by catching far more
of the real upsells (recall) at a similar precision.

**Recommendation:** replace any manual "LTV above X, CAC below Y" outreach
rule with the model's probability score. The gap between the model's recall
(0.86) and the rule's (~0.27–0.72 depending on version) is large enough that
a rule-based approach is leaving real upsell revenue unflagged.

### 3. Who will become a "super customer"?

**Model:** CatBoost classification on `referred`, scored 0–100. Two versions
were built and compared: a v1 using all available data (Accuracy 0.824, F1
0.761, ROC-AUC 0.867) and a v2 restricted to information genuinely known at
acquisition time — no `upsell`, `ltv_months`, or `cumulative_profit` (Accuracy
0.755, F1 0.720, ROC-AUC 0.817).

**Finding:** v1's top features (`upsell`, `ltv_months`, `purchased` — ~86% of
its decision weight) are themselves late-relationship outcomes, which makes v1
excellent at describing *already-mature* super customers but unusable for
scoring a brand-new customer on day one. v2 pays a real but modest accuracy
cost to fix this, and its score distribution is far less extreme (only 2% of
test customers land in the "High" tier vs. 32% in v1) — v1's tier thresholds
do not transfer to v2 without recalibration.

**Recommendation:** use v2 for any real day-one scoring use case; keep v1 only
as a retrospective/loyalty-program targeting tool for customers who are
already established. Recalibrate v2's Low/Medium/High thresholds against its
own score distribution before using them operationally.

### 4. Where should the ad budget go to maximize profit?

**Model:** LightGBM regression on `cumulative_profit` (holdout R² ≈ 0.75, CV
Mean R² ≈ 0.76), used to simulate reallocating the ₪50,000/month budget across
~11 sampled "campaign slot" profiles.

**Finding:** `ad_budget` ranks **last** of all 15 features in importance. A
simulated +20% budget increase produced **zero** additional predicted profit;
a profit-weighted reallocation across profiles actually **underperformed**
simple equal allocation. Once lead volume and follow-up execution are known,
raw ad spend adds almost nothing further to the profit prediction. A second,
independent check confirms this: spreading the same total budget across more
campaigns doesn't improve efficiency either — predicted profit **per
campaign** is roughly flat (~₪13,000–14,300) from 5 through 50 campaigns.
Total predicted profit keeps climbing with campaign count only because it's
a sum over more customer profiles, not because each campaign is performing
better.

**Recommendation:** do not increase ad spend expecting more profit on its
own, and don't treat "more campaigns" as inherently more efficient either —
this model finds no evidence that either lever helps here. Redirect that
investment toward increasing lead volume and improving follow-up execution,
the two clusters of features that actually drive predicted profit. Treat any
real-world budget change as a hypothesis to validate with a controlled pilot,
not a conclusion to act on directly from this simulation (the dataset has no
campaign/channel identifier, so this is a same-profile budget-varying
simulation, not a true multi-channel allocation test).

### 5. Are late follow-ups a waste of time?

**Method:** stage-by-stage lead-retention/dropout curve across the five
follow-up rounds, plus `calls_to_closed` vs. `calls_to_not_closed` as an
aggregate effort comparison (the dataset has no column linking a specific
closed deal to a specific follow-up stage, so no per-stage close count is
invented).

**Finding:** the sales manager is largely wrong. Drop-off does not decline
smoothly — **Follow-up 4 has the lowest drop-off of any stage (10.4%)**, while
**Follow-up 5 has the highest (29.2%)**. Tens of thousands of leads are still
active well past the 3rd call (46,323 / 41,517 / 29,384 remaining at
Follow-ups 3/4/5, against 10,557 total closed deals). Closed deals also
average slightly *fewer* calls than abandoned ones (3.52 vs. 3.93) — dragging
a deal out is mildly associated with a lower chance of closing, but that's a
different finding from "stop after call 3."

**Recommendation:** keep the full 5-call sequence — cutting off after call 3
would forfeit Follow-up 4, the single strongest-retaining stage in the funnel.
Instead, investigate Follow-up 5 specifically (script, timing, lead fatigue),
since that's where the real, isolated problem is. Use a rising call count past
~4 calls as a soft prioritization signal, not a hard cutoff policy.

## Cross-cutting: data leakage discipline

A project-wide audit checked every package for "late-relationship" columns
(`ltv_months`, `upsell`, `referred`, `cumulative_profit`) being used as
features for the wrong target — using an outcome to predict another outcome
inflates offline metrics while making the model useless in production, since
that information isn't actually available at prediction time. Two real gaps
were found and fixed:

- **Package 3** (upsell) was retrained without `ltv_months` — a ~1 F1-point
  cost, more than offset by the resulting model actually being deployable.
- **Package 4** (super customer) was rebuilt as a v2 without `upsell`,
  `ltv_months`, or `cumulative_profit` — a larger (~4-point) but necessary
  cost, since v1's top three features were exactly this category of leakage.
- **Package 2** (lifetime) had `upsell` removed too, though its measured
  impact was negligible (~0.4% importance) — fixed for consistency rather
  than because it changed the result.

The general rule applied throughout: whichever of these four columns isn't
the current package's own target must be justified as genuinely available at
prediction time, or excluded.

## Package 1 — data quality, in brief

3,500 raw rows → 33 missing values across 2 columns (imputed), 10 exact
duplicates removed, 0 business-logic violations found. `ad_budget` and
`num_leads` are strongly correlated (Pearson r ≈ 0.98), but leads-per-shekel
drops sharply from the bottom to the top budget quartile — the same
diminishing-returns signal that shows up independently in Package 6's finding
that ad spend barely moves predicted profit.

## Bottom line for the founder

The clearest, most consistent theme across all six packages: **spend and
volume alone don't move the numbers that matter — execution does.** Ad budget
barely affects profit; follow-up execution (specifically fixing whatever is
wrong at the 5th call) affects retention far more than call-count policy
would suggest; and the fastest lever for both upsell and super-customer
targeting is using the models built here instead of manual threshold rules,
which consistently and substantially underperform them.
