# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Humais Ali
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** [https://github.com/humaisali/FlyRank-Machine-Learning-Internship](https://github.com/humaisali/FlyRank-Machine-Learning-Internship)
- **Date:** October 2026

> This report documents the completed Search Intelligence & Content Opportunity Scoring Capstone. Sections 1–8 mirror the evaluation rubric axes, while Sections 0 and 9 represent canonical research paper sections mirrored in the public web release.

---

## 0. Abstract

Can historical search performance, content freshness, engagement metrics, and visibility signals prioritize content pages that need optimization before performance decays? Using an anonymized portfolio of 30,000 published pages across 32 enterprise clients from the FlyRank Search Intelligence dataset, we engineered a leakage-safe feature space spanning search demand, user engagement, and content age. We evaluated heuristic baseline rules alongside supervised classification and continuous regression models under strict client-holdout cross-validation to prevent inter-client leakage. In out-of-sample client-holdout validation, our Random Forest classifier achieved a Precision@50 of 0.740 (a 3.08× lift over the business rule baseline of 0.240) and an ROC AUC of 0.750, while continuous trend regression revealed that historical signals alone do not reliably predict continuous percentage changes ($R^2 \approx -0.01$ to $-0.24$). We synthesized these findings into a transparent decision-support action playbook that assigns tiered interventions (refresh immediately, review CTR, review engagement, monitor) with verifiable reason codes, giving editorial teams an evidence-backed queue for resource allocation.

---

## 1. Problem framing

### The Decision Supported
Publishers, digital media networks, and enterprise SEO teams maintain vast catalogs of thousands or tens of thousands of published articles. Editorial and technical optimization resources are strictly finite: teams can realistically audit and refresh only 20 to 50 URLs each week. Historically, teams prioritize pages either by intuition, raw traffic volume, or naive keyword ranking checks. This capstone supports a specific operational decision: **Which published URLs should an editorial team review and refresh this sprint to recover decaying visibility and capture unharvested engagement?**

### Unit of Analysis & System Output
- **Unit of Analysis:** A single published content URL/page observed over a trailing 90-day performance window.
- **System Output:** A calibrated **Opportunity Score (0–100)**, an assigned **Action Tier** (`refresh`, `refresh_and_review_ctr`, `refresh_and_review_engagement`, `expand_and_refresh`, `monitor`), and transparent **Reason Codes** explaining exactly why the page was flagged.

### Human Action & The Cost of a Wrong Call
- **Human Action:** An editor inspects the top-ranked rows in the queue, assesses search intent shifts, updates outdated statistics or examples, optimizes metadata (titles/meta-descriptions for CTR issues), or improves internal linking.
- **Cost of False Positives:** Wasted editorial hours rewriting stable or naturally cyclical evergreen pages that did not require intervention.
- **Cost of False Negatives:** Silently declining high-value pages that lose page-1 rankings, costing organic impressions and qualified sessions that take months to recover.
- **Why Machine Learning Helps:** A fixed rule (e.g., "refresh all pages older than 180 days with >1,000 impressions") triggers massive false-positive volume on evergreen content and completely misses decaying mid-tail pages. Supervised ranking surfaces non-linear combinations of impression volume, position drift, and engagement decay that heuristic rules cannot isolate.

---

## 2. Data safety & Contract

### Data Provenance
We utilized the FlyRank Search Intelligence dataset (`data/raw/content_refresh_anonymized.csv`), comprising 30,000 pseudonymized content rows across 32 enterprise clients over a 90-day observation window.

### Excluded Columns & Leakage Prevention
To ensure strict public safety and prevent fatal target leakage, the following variables were deliberately excluded:
1. **Target Leakage Fields:** `trend_direction` and `trend_pct`. Because the primary classification outcome (`is_declining_label`) is derived directly from `trend_direction == "down"`, including either variable as a feature produces 100% artificial accuracy and catastrophic production failure.
2. **Pseudonymous Identifiers:** `content_id` and `client_id` were stripped from the feature matrix. Client IDs were preserved strictly as grouping variables for client-holdout cross-validation splits.
3. **Sensitive & Proprietary Entities:** No raw URLs, article titles, domain names, raw query strings, or client names exist in the dataset or pipeline.

### Handling Domain Nuances & Gotchas
- **Percentage Scaling:** All rate columns (`ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `trend_pct`) are pre-scaled percentage values where $1.0 = 1\%$, not $100\%$.
- **Position Zero Sentinel:** `avg_position = 0` denotes missing ranking data (no impressions recorded), not a rank of zero. Missing positions were handled via median imputation and flagged with indicator flags.
- **Measurement Systems Discrepancies:** `scroll_rate` and `ai_traffic_pct` can exceed 100 due to differing tracking denominators across client tagging layers; values were preserved without truncation to maintain signal integrity.

---

## 3. Baseline Design

Before training machine learning algorithms, we formulated a transparent business heuristic baseline mirroring industry-standard SEO logic:

$$\text{Baseline Score} = 0.40 \times \text{rank}(\text{search\_volume}) + 0.30 \times (1 - \text{rank}(\text{avg\_position})) + 0.30 \times (1 - \text{rank}(\text{days\_since\_last\_update}))$$

The baseline prioritizes stale articles with high search demand ranking on competitive positions. When evaluated on the holdout client partition:
- **Baseline Precision@50:** **0.240** (only 12 of the top 50 flagged pages were actually in decline).
- **Baseline ROC AUC:** **0.627**.
- **Continuous Baseline $R^2$:** **-0.719** ($MAE = 282.74$).

This simple heuristic fails because high search volume and older content age frequently correlate with high authority and stable rankings. A rule that ignores engagement velocity and multi-day impression persistence produces high false-alarm rates.

---

## 4. Model Architecture & Training

We approached the opportunity scoring challenge from two complementary paradigms:

### Paradigm A: Supervised Classification for Refresh Prioritization (Primary)
- **Target:** `is_declining_label` ($\text{trend\_direction} == \text{"down"}$), with an empirical base rate of **54.2%** (16,262 declining items out of 30,000).
- **Candidate Models:**
  1. Regularized Logistic Regression (L2 penalty, StandardScaled)
  2. Decision Tree Classifier ($\max\_depth = 6$)
  3. Random Forest Classifier (200 estimators, balanced subsample, out-of-bag scoring)
- **Features (22 Total):** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `impressions_90d`, `clicks_90d`, `sessions_90d`, `engaged_sessions_90d`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `days_with_impressions`, `days_with_sessions`, plus missingness indicator flags.

### Paradigm B: Continuous Trend Regression (Negative Result Audit)
- **Target:** `trend_pct` (continuous percentage momentum).
- **Candidate Models:** Dummy Mean Regressor, OLS Linear Regression, Random Forest Regressor.

---

## 5. Evaluation & Validation Design

### Split Strategy: Grouped Client Holdout
Standard random k-fold cross-validation is invalid for multi-tenant search data because content published under the same client domain shares site-wide domain authority, technical infrastructure, and CMS templates. Random splits leak site-level variance across train and test sets, inflating apparent performance.

We enforced a **Client Holdout Split**, reserving ~20% of clients (6 complete client portfolios, 5,892 pages) exclusively for out-of-sample evaluation. No client appeared in both training and test partitions.

### Quantitative Results Table

#### 1. Classification Performance (Primary Decision Metric)
*Base Rate: 54.2%*

| Model | ROC AUC | Avg Precision | Precision@50 | Recall | F1 Score | Lift over Baseline |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0.750** | **0.618** | **0.740** | **0.744** | **0.640** | **+208% (3.08×)** |
| Decision Tree | 0.742 | 0.575 | 0.540 | 0.716 | 0.634 | +125% (2.25×) |
| Logistic Regression | 0.700 | 0.522 | 0.400 | 0.567 | 0.566 | +66.7% (1.67×) |
| **Heuristic Baseline** | **0.627** | **0.468** | **0.240** | — | — | Baseline |

#### 2. Continuous Trend Regression (Audit on Test Split)

| Model | MAE | RMSE | $R^2$ Score | Interpretation |
|---|---:|---:|---:|---|
| Dummy Regressor (Mean) | 64.16 | 315.97 | -0.0001 | Uninformative mean baseline |
| Linear Regression | 65.87 | 317.03 | -0.0068 | Historical features lack linear momentum signal |
| Random Forest Regressor | 72.11 | 351.91 | -0.2405 | Non-linear regression overfits extreme volatility |
| Business Baseline | 282.74 | 414.30 | -0.7194 | Heuristic rankings deviate severely from trend |

### Error Analysis
- **False Positives in Precision@50:** The Random Forest's 13 non-declining picks in the top 50 (26%) consisted predominantly of mature pages in competitive verticals with volatile impression swings but stable overall click-through rates.
- **Linear Separation Failure:** As shown by the regression results, predicting the exact *magnitude* of future percentage change from 90-day aggregate snapshot data is computationally infeasible due to external SERP dynamics and algorithmic volatility. However, predicting the *binary directional propensity* (`is_declining_label`) achieves strong, actionable ranking precision (Precision@50 = 0.740).

---

## 6. Interpretation & Findings

### Top Predictive Signals
Feature importance extracted from the calibrated Random Forest reveals where signal resides:
1. `days_with_impressions` (15.78%): The regularity and persistence of Google search impressions across the 90-day window is the strongest single indicator of stability vs decay. Pages losing daily impression presence are entering algorithmic decline.
2. `log_impressions_90d` (12.82%): Aggregate search visibility scales non-linearly with refresh payoff.
3. `avg_position` (10.90%): Pages hovering between positions 8 and 20 represent peak opportunity; pages beyond position 50 rarely respond to on-page refresh alone.
4. `content_age_days` / `days_since_last_update` (9.55%): Freshness is an amplifier, not an independent driver.
5. `char_count` & `word_count` (8.23% combined): Thin content correlates with vulnerability to organic erosion.
6. `ctr` (3.30%) & `scroll_rate` (3.11%): Discrepancies between high visibility and low engagement identify prime candidates for metadata and UX restructuring.

### Key Surprises & Negative Findings
- **Keyword Demand is Not Quality:** Raw search volume alone exhibited near-zero correlation with decay. High-volume terms frequently suffer fierce SERP feature crowding (video carousels, AI Overviews) that decouple organic ranks from actual clicks.
- **The Limits of Continuous Forecasting:** We proved empirically that search engine performance momentum cannot be forecast as an unconstrained continuous variable ($R^2 < 0$). Real-world search data requires ranking / decision-support framing rather than point forecasting.

---

## 7. Ranked Recommendations Playbook

The final scoring engine blends calibrated model probability with search demand and operational flags into a normalized 0–100 **Opportunity Score**:

$$\text{Opportunity Score} = 0.50 \times (\hat{P}_{\text{decline}} \times 100) + 0.30 \times \text{percentile}(\log(\text{impressions})) + 0.20 \times \text{gap}_{\text{CTR}}$$

### Editorial Action Matrix

| Action Tier | Count in Portfolio | Target Characteristics | Prescribed Editorial Playbook |
|---|---:|---|---|
| **Refresh & Review CTR** | 6,657 (22.2%) | High impressions, low CTR, declining trend | Rewrite title tags and meta descriptions; optimize headline search intent; audit SERP snippet. |
| **Refresh & Review Engagement** | 1,990 (6.6%) | Declining trend, poor scroll rate (<30%), low session duration | Update outdated statistics; add visual media/tables; restructure introduction; improve UX. |
| **Refresh Immediately** | 8,178 (27.3%) | High model decline probability (>75%), high historical demand | Full editorial refresh: expand thin sections, add expert quotes, fix broken links, rebuild internal links. |
| **Expand & Refresh** | 82 (0.3%) | High rankings, thin content (<500 words), dropping positions | Significantly expand word count; add comprehensive FAQs; answer long-tail queries. |
| **Monitor** | 13,093 (43.6%) | Stable or growing performance, healthy engagement | Do not touch. Prevent editorial resources from disrupting stable ranking signals. |

### Top 5 Actionable Queue Preview (Public-Safe)

| Rank | Score | Model Prob | Prescribed Action | Primary Reason Codes | Imp 90d | Sessions |
|---:|---:|---:|---|---|---:|---:|
| 1 | 81.6 | 0.782 | `refresh_and_review_ctr` | `declining_with_demand`, `low_ctr_visible_page`, `model_decline_risk` | 12,834 | 66 |
| 2 | 81.4 | 0.788 | `refresh_and_review_ctr` | `declining_with_demand`, `low_ctr_visible_page`, `visible_model_opportunity` | 8,064 | 23 |
| 3 | 81.4 | 0.847 | `refresh_and_review_ctr` | `declining_with_demand`, `low_ctr_visible_page`, `ctr_review_candidate` | 2,498 | 9 |
| 4 | 81.0 | 0.774 | `refresh_and_review_ctr` | `declining_with_demand`, `low_ctr_visible_page`, `model_decline_risk` | 13,790 | 27 |
| 5 | 80.9 | 0.815 | `refresh_and_review_ctr` | `declining_with_demand`, `low_ctr_visible_page`, `visible_model_opportunity` | 3,393 | 5 |

---

## 8. Reproducibility & Environment

### Reproduction Commands
To replicate all findings from a fresh git clone:
```bash
# 1. Clone repository
git clone https://github.com/humaisali/FlyRank-Machine-Learning-Internship.git
cd FlyRank-Machine-Learning-Internship

# 2. Set up virtual environment and dependencies
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Execute reference pipeline (Stages 1 through 5)
python scripts/run_all.py

# 4. Execute full Capstone notebook top to bottom
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb --inplace
```

### Random Seeds & Environment Parameters
- **Global Random Seed:** `random_state = 42` across all train/test splits, decision trees, and random forests.
- **Python Version:** 3.11 / 3.12 / 3.14 compatible.
- **Primary Dependencies:** `scikit-learn>=1.4`, `pandas>=2.2`, `numpy>=1.26`, `matplotlib>=3.8`, `reportlab>=4.0`.
- **Sealed Artifacts:**
  - Pipeline summary: `outputs/model_results.json`
  - Model report: `outputs/model_report.md`
  - Ranked recommendations: `outputs/refresh_queue.csv` and `work/outputs/final_ranked_recommendations.csv`

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**.

We express our gratitude to the FlyRank team for curating and hosting the anonymized enterprise search dataset that made this research possible. For more information regarding FlyRank's search intelligence and discoverability platform, visit [https://flyrank.ai](https://flyrank.ai).
