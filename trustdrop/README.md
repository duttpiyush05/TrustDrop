# TrustDrop — Trustworthy AI-Based Student Dropout Prediction and Counseling System

> **Research / educational decision-support prototype. Not production-ready.** Predictions must never be used to
> punish students, deny opportunities, label students automatically, or make final decisions without human review.

##  Synthetic / development-data warning (read first)

The real UCI dataset was **not available** while this project was built (the build sandbox could not reach it).
Everything in `reports/`, `models/best_model.joblib` and the demo students in the app was produced from a
**synthetic development dataset** (`data/synthetic_dev.csv`: random data with a weak planted signal). It exists
only to verify that the pipeline, trust modules, API and UI work. **None of those numbers are project accuracy or
real-world performance**, and no conclusion about real students can be drawn from them. Every report, API response and
UI page carries a `provenance` label and the UI shows a "Development / synthetic data" banner. The synthetic file is used
**only** when you pass `--allow-synthetic`; it is never substituted silently. Training on the real data overwrites all of it.

## 1. Project overview
TrustDrop predicts whether a student is likely to **Dropout**, remain **Enrolled** or **Graduate**, and, more importantly,
wraps every prediction in a *Trustworthiness Layer*: it asks not only "is it accurate?" but whether predictions are
explainable, calibrated, uncertainty-aware, fair, robust, based on valid data, and appropriate for human decision-making.

## 2. Problem statement
Universities want to spot students who may need support early. A black-box score is risky: it can be overconfident,
unfair to groups, fragile, wrong on unusual or invalid data, and easy to over-trust.

## 3. Existing / base system
A conventional ML pipeline on the UCI *Predict Students' Dropout and Academic Success* data (4,424 students, 36 predictors +
target) compares Logistic Regression, Random Forest, XGBoost and an ANN, and reports a dashboard-style risk output. The
reference work reports XGBoost at roughly 85% accuracy; **we do not reuse that figure**. Our system trains and evaluates
everything itself and selects the best model by its own criterion.

## 4. Our Trustworthy-AI contribution
Explainability (SHAP) · uncertainty (entropy, margin) · calibration (Brier, ECE, reliability) · fairness audit (Fairlearn) ·
robustness testing · data-quality/OOD validation · a project-defined trustworthiness indicator · human-in-the-loop counselor review.

## 5. Architecture
```
React (TypeScript, Vite, Tailwind)
   │ HTTP/JSON
FastAPI routes (thin)  ──►  services/ (TrustEngine, repository)  ──►  SQLite (SQLAlchemy)
                                   │
                          ml/  ┌ data_quality (validate + OOD)
                               ├ prediction (calibrated model, risk bands)
                               ├ uncertainty · explainability (SHAP) · robustness
                               ├ fairness (system audit) · calibration
                               └ trust/indicator
Student data → Validation → Prediction → Probabilities → Uncertainty → Explanation → Robustness
             → Data quality/OOD → (system fairness + calibration) → Trust indicator → Human review → Counseling action
```
Training (`scripts/train.py`) writes `models/best_model.joblib` (model + preprocessing + trust reference data) and `reports/*`.

## 6. Data description
UCI dataset 697. 36 predictors: 9 nominal codes (course, application mode, parents' qualification/occupation, …) →
one-hot; 8 binary flags; 19 numeric (grades, curricular-unit counts per semester, macro-economic indicators). Target:
Dropout / Enrolled / Graduate (imbalanced). The schema lives in `ml/preprocessing/schema.py` (single source of truth).
Note: the data has no attendance column; robustness perturbs grades and unit counts instead.

## 7. ML models
Logistic Regression, Random Forest, XGBoost, ANN (MLP 64-32, early stopping). Each is one sklearn `Pipeline`
(preprocessing + model), seed 42. Imbalance: balanced class weights (LR, RF) or balanced sample weights computed inside each
`fit` (XGBoost, ANN).

## 8. Evaluation methodology
Stratified 80/20 train/test split. Stratified 5-fold cross-validation on the **training split only**, identical for all
models, re-fitting preprocessing in every fold (no leakage). Metrics: accuracy, balanced accuracy, macro precision/recall/F1,
weighted F1, one-vs-rest macro ROC-AUC, log-loss. **Model selection = highest mean CV macro-F1** (tie → ROC-AUC). The test set is
used only for reporting. Confusion matrices per model are saved as JSON. A test scrambles test-set labels and verifies that
CV results, selection, calibration and the saved model do not change.

## 9. Explainability
SHAP computed on the real fitted (uncalibrated) base pipeline: TreeExplainer (XGBoost/RF), LinearExplainer (LR),
KernelExplainer (ANN). One-hot columns are summed back to the 36 original features (SHAP is additive). Per student: top
contributions for each class, with signs; global importance = mean |SHAP| on a training sample. Tests check *local accuracy*
(base value + Σ contributions equals the model's own raw output). If SHAP fails, the API returns `available: false` plus the
error; nothing else breaks. SHAP describes model behaviour, not causes.

## 10. Uncertainty
For probabilities p over K=3 classes: confidence = max p; margin = p₍₁₎ − p₍₂₎; entropy H = −Σ p ln p, normalised by ln K.
**HIGH** if normalised entropy > 0.85 or margin < 0.15; **LOW** if entropy ≤ 0.65 and margin ≥ 0.40; else MEDIUM
(project-defined heuristics). HIGH ⇒ human review. Also reported: disagreement between the cross-fitted calibration-ensemble
members (epistemic proxy). These measure how *decided* the model is; they are **not** the probability of being correct.

## 11. Fairness
Audited decision: "model flags a student as Dropout". Attributes available in the data: gender, age group (≤20, 21–25, ≥26),
international, special needs, displaced. Per group (Fairlearn `MetricFrame`): n, actual dropout rate, flag rate, precision,
recall, FPR, FNR, accuracy. Gaps: demographic parity, equal opportunity (recall), FPR, equalized odds. Status by worst gap:
< 0.10 *Fair*, < 0.20 *Potential disparity*, else *Requires review*. Groups with n < 30 are excluded and noted. Evaluated on
held-out data only. Gaps may reflect real base-rate differences and do **not** prove bias; gender and age are also model inputs.
No mitigation is implemented (see Future work).

## 12. Robustness
Per student: 30 (configurable) perturbed copies: grades get Gaussian noise (0.10 × training std), unit counts ±1, clipped to valid
ranges, approved ≤ enrolled + credited enforced, demographics untouched. Stability = share keeping the same class; also mean/max
change in the predicted class' probability. Score = stability × (1 − min(1, mean shift/0.20)). *Stable*: ≥90% same and max shift ≤ 0.15;
*Unstable*: <70% or max shift > 0.30; else *Sensitive* (warning shown). Seeded, so reproducible. System level: same test on 100 held-out rows.

## 13. Calibration
Metrics: multiclass Brier score, top-label ECE (10 bins), reliability-curve data and plot (`reports/calibration/`). The method
(none / Platt-sigmoid / isotonic via `CalibratedClassifierCV`) is chosen by **out-of-fold Brier on training data**; calibration is applied
only if it strictly improves it. Test data only reports before/after. Balanced class weighting deliberately shifts probabilities, which is one
reason uncalibrated models can look miscalibrated.

## 14. Data quality / OOD
**Invalid** (model not run): missing, non-numeric, non-integer where required, out-of-range (e.g. admission grade must be 0–200),
bad binary/category values, impossible combinations (approved units > enrolled + credited; relaxed to a warning if >0.5% of training rows
violate it). **Unusual** (valid, flagged): unseen category code, outside training min/max, robust z-score > 6, or Mahalanobis distance
(Ledoit-Wolf covariance, fitted on training data) above the 99.5th percentile of training distances. **Normal** otherwise. This is separate from
model uncertainty: it says "is the input reliable?", not "is the model torn?".

## 15. Trustworthiness Indicator (project-defined)
**Not an established scientific or industry metric.** Six components in [0,1]: explainability (SHAP produced: 1/0, availability only, weight 0.10),
uncertainty (1 − normalised entropy, 0.20), calibration (1 − min(1, ECE/0.15), 0.15), fairness (1 − min(1, worst equalized-odds gap/0.30), 0.20),
robustness (score above, 0.20), data quality (normal 1.0, unusual 0.5, 0.15). Indicator = weighted mean of the **evaluated** components (weights renormalised);
unevaluated components show "Not evaluated" and are never invented; invalid input gets no indicator. Levels: ≥0.75 High, ≥0.50 Moderate, else Low;
if any component < 0.40 the level is capped at Moderate. Two scopes: per prediction, and system level (held-out evaluation).
Limitations: subjective weights/thresholds, components on different scales, no guarantee of correctness or fairness.

## 16. Human-in-the-loop counseling
Wording is always "The model estimates a … probability of dropout based on the available data" — never "will drop out". Risk bands from P(Dropout):
≥0.50 high, ≥0.30 medium. "Human counselor review recommended" triggers on high risk, high uncertainty, unusual input or sensitive predictions.
Counselors see prediction, explanation, uncertainty and trust info; they **accept** or **override** (decision + reason required), add notes and schedule
counseling. All stored; the final decision is the counselor's.

## 17. Backend architecture
`backend/app/`: `main.py` (app factory, CORS, lifespan, error handler) · `api/` (thin routes) · `schemas/` (Pydantic) · `services/` (`assessment.TrustEngine`,
`repository`, `seed`, `reports`) · `database/` (SQLAlchemy models/session) · `ml/` (all ML and trust logic). Routes contain no ML code. Logs never include student data.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `/model/info`, `/model/performance`, `/schema/features` | system, model, evaluation, form schema |
| `POST /validate` | data-quality check only |
| `POST /predict` | full pipeline for `{student_id}` or `{features}`; stores the result; `422 INVALID INPUT` if invalid |
| `POST /explain`, `POST /trust/robustness` | SHAP explanation; perturbation test |
| `GET /trust/summary`, `/trust/fairness`, `/trust/calibration`, `/trust/explainability` | system-level trust artifacts |
| `GET /students`, `/students/{id}`, `/predictions`, `/predictions/{id}` | demo students and stored assessments |
| `POST /counseling/review`, `GET /counseling/reviews`, `POST /counseling/schedule`, `GET /counseling/schedules` | human review |
| `GET /dashboard/overview` | dashboard data |

Interactive docs at `http://localhost:8000/docs`. Response bodies of `/predict` and the trust endpoints are structured JSON (typed on the frontend); request bodies are Pydantic-validated.

## 18. Frontend architecture
React 18 + TypeScript + Vite + Tailwind 4. `services/api.ts` (typed API layer), `hooks/useApi.ts` (loading/error/empty states), `context/` (latest assessment),
`components/ui.tsx` (cards, badges with text+symbol so colour is never the only signal, charts as plain SVG/CSS — no chart library), `pages/`: Overview, Student prediction,
Trust report, Explainability, Fairness, Model performance, Robustness, Counselor review.

## 19. Database
SQLite (`data/trustdrop.db`, created automatically): `students` (synthetic names + features), `predictions` (full trust report JSON), `reviews`, `schedules`.
Demo students are seeded from held-out dataset rows on first start.

## 20–23. Installation, running, tests
```bash
# Backend (Python 3.10+)
pip install -r backend/requirements.txt
python scripts/make_synthetic_dev_data.py        # ONLY if you have no real data (development)
python scripts/train.py --allow-synthetic        # development; or: python scripts/train.py  (real data/data.csv)
cd backend && python -m uvicorn app.main:app --port 8000

# Frontend (Node 18+), in another terminal
cd frontend && npm install && npm run dev        # http://localhost:5173  (API URL: VITE_API_URL, default http://localhost:8000)
npm run build                                    # type-check + production build

# Tests
cd backend && python -m pytest
```
Config via `TRUSTDROP_*` environment variables (`database_url`, `cors_origins`, `n_perturbations`, `random_seed`, directories).

## 24. Real dataset instructions
Download `data.csv` from https://archive.ics.uci.edu/dataset/697 into `data/data.csv` (semicolon-separated, original headers; handled automatically, tested with a simulated raw file).
Run `python scripts/train.py`, delete `data/trustdrop.db` to re-seed demo students from the real data, restart the API. No code changes needed; banners disappear automatically.

## 25. Limitations
Only the synthetic pipeline has been verified so far. One dataset from one institution; limited generalisation. No attendance data. Calibration and fairness depend on the test split
size (small groups are excluded). SHAP explains the model, not causality. Thresholds (uncertainty, risk, fairness, robustness, OOD, indicator) are heuristics. OOD uses numeric features only.
Sensitive attributes are model inputs. Demo seeding uses held-out rows, so demo students are not a prospective cohort. Response schemas for the large trust report are not fully modelled in Pydantic.

## 26. Ethical considerations
Student data is sensitive: use synthetic names, don't put personal details in notes, no student values in logs. The system supports counselors; it must not penalise, label or exclude students.
Risk estimates can be wrong and can reflect historical inequalities. Override reasons are recorded to keep humans accountable.

## 27. Future work
Fairness mitigation (e.g. Fairlearn ThresholdOptimizer) with before/after trade-offs; conformal prediction sets; temporal validation; subgroup calibration; authentication and audit logging; richer
counseling workflow; full Pydantic response models; evaluation on the real dataset.
