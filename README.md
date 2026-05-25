# Criteo Uplift Report Reproducibility

This repository reproduces the figures and core metrics summarizedby running the notebook `code/Qingwei_Prinu - Causal Models - Final Project Report.ipynb` end to end.

The notebook is the executable source of truth for implementation details. The notes HTML is the benchmark reference for expected results and interpretation.

## Reproduction Scope

- Primary outcome: `conversion`
- Secondary sensitivity outcome: `visit`
- Main effect target: ITT/ATE from randomized assignment (`treatment`)
- Heterogeneity target: CATE/uplift ranking across `T`, `X`, `R`, `DML`, and `DR` learners

## Files Used for Reproduction

- Notebook (run this): `code/Qingwei_Prinu - Causal Models - Final Project Report.ipynb`
- Environment file: `code/causal-inference-program.yml`
- Local toolkit package: `code/causal_toolkit/`

## 1) Create the Environment

From the repository root:

```powershell
conda env create -f "code/causal-inference-program.yml"
conda activate Causal-Inference-Program-Env
pip install scikit-learn
```

Why `scikit-learn` is included explicitly here: the notebook imports sklearn modules directly (`LogisticRegression`, `train_test_split`, `MiniBatchKMeans`) and this package is not pinned explicitly in the YAML.

## 2) Download and Place the Dataset

1. Download the Criteo Uplift v2.1 dataset.
2. Place the CSV at:

   `code/data/criteo-uplift-v2.1.csv`

3. Confirm the CSV contains:
   - Features: `f0` through `f11`
   - Treatment/pathway/outcomes: `treatment`, `exposure`, `visit`, `conversion`

Note: `code/data/*.csv` is ignored by git (`.gitignore`), so the data file is intentionally not committed.

## 3) Run the Notebook in Order

Run from the `code/` directory so all relative paths/imports resolve correctly (`./data`, `./figures/final_report`, `causal_toolkit`):

```powershell
cd code
jupyter lab "Qingwei_Prinu - Causal Models - Final Project Report.ipynb"
```

Then run all cells from top to bottom without reordering.

## 4) Key Notebook Anchors (Should Match)

The notebook constants used for the reported pipeline include:

- `data_path = Path("./data/criteo-uplift-v2.1.csv")`
- `heterogeneity_sample_size = 300_000`
- `heterogeneity_test_size = 0.40`
- `mediation_sample_size = 250_000`
- `refutation_sample_size = min(120_000, len(criteo_uplift_df))`

Important seeds used in major sections:

- Primary heterogeneity run: `random_state=42`
- Visit sensitivity heterogeneity run: `random_state=43`
- DoWhy refutation sample: `random_state=2026`

## 5) Expected Output Artifacts

The notebook writes report figures to `code/figures/final_report/`, including:

- `figure_1_overall_rates.png`
- `figure_2_conversion_rates_by_group.png`
- `figure_2a_covariate_imbalance_before.png`
- `figure_2b_covariate_balance.png`
- `figure_2c_exposure_compliance.png`
- `figure_3_average_treatment_effects.png`
- `figure_4a_uplift_score_distribution.png`
- `figure_4b_ranked_uplift_comparison.png`
- `figure_4c_model_level_cate_vs_top_quintile_lift.png`
- `figure_5a_treatment_attenuation.png`
- `figure_5b_pathway_odds_ratios.png`
- `figure_6a_t_learner_ite_segmentation.png`
- `figure_6b_x_learner_ite_segmentation.png`
- `figure_6c_r_learner_ite_segmentation.png`
- `figure_6d_dml_ite_segmentation.png`
- `figure_6e_dr_learner_ite_segmentation.png`
- `figure_6d_cross_model_segment_heatmaps.png`
- `figure_7a_qini_curve_primary_outcome.png`
- `figure_7b_auuc_model_ranking.png`

## 6) Verification Checkpoints

Use the following checkpoints to verify you reproduced the same analysis profile reported in the notes:

1. Data load profile:
   - Rows near `13,979,592`
   - Rates near: treatment `85.00%`, exposure `3.06%`, visit `4.70%`, conversion `0.29%`
2. ATE values (full dataset):
   - `visit` ATE about `+1.0342` percentage points (95% CI about `[1.0056, 1.0629]`)
   - `conversion` ATE about `+0.1152` percentage points (95% CI about `[0.1085, 0.1219]`)
3. AUUC ranking (conversion holdout):
   - `T-learner` (`100.1273`) > `DR-learner` (`96.4793`) > `X-learner` (`91.3866`) > `DML` (`86.1768`) > `R-learner` (`84.5725`)
4. DoWhy snapshot:
   - Baseline estimated effect around `0.0009` on the refutation sample, with stable placebo/subset/random-common-cause checks

## 7) Reproducibility Notes and Known Variance

- Exact byte-for-byte figure and metric matches are not guaranteed across environments.
- Small variation is expected in some CATE/uplift outputs because several LightGBM-based learner paths rely on model defaults.
- Core descriptive rates and ATE estimates should remain very close when the same data and execution order are used.
- If DoWhy is missing, the notebook prints an install hint (`pip install dowhy`) and marks that section as skipped.

## 8) Recommended Run Order for Reliable Reproduction

1. Create environment.
2. Place dataset at the expected path.
3. Launch notebook from `code/`.
4. Run all cells once, top to bottom.
5. Compare produced figures and checkpoints against `code/causal_inference_criteo_uplift_final_notes.html`.

