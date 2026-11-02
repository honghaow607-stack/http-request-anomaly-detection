# Matched representation baseline study

This supplementary study was actually fitted on the supplied corpus during the manuscript revision. It is separate from the original supplied experiment archive. The protocol was fixed locally before fitting; it is not an external preregistration.

## Evidence and scope

Five configurations, five seeds and 25 model fits share the same parsed/deduplicated 26,646 requests and 64/16/20 stratified splits. Every configuration uses balanced 200-tree RF, the seed as random_state, max_features=sqrt, and the estimator predict decision rule. All three hybrid encodings realize 5,040 features. Numerical-only has 40 and word-text-only has 5,000. The vocabulary is fitted only on training records. No model, threshold or baseline is selected using the new test results.

The joined-word representation is motivated by [Song et al.](https://doi.org/10.9708/jksci.2026.31.01.109), who used joined HTTP text and word n-grams. The character alternative is motivated by [Shaik et al.](https://doi.org/10.5539/cis.v19n2p124), who studied field-specific character-3-gram encodings with metadata. These are controlled adaptations using the present target/body fields, decoder, numerical block and RF, not full reproductions of the published systems. Their published scores are not placed in the matched table.

The full protocol is `results/representation_baselines/protocol.md`; the runner is `scripts/run_representation_baselines.py`. Python 3.12.14, scikit-learn 1.3.2, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.11.4 were used. The direct/transitive runtime requirements are pinned in `docs/requirements-representation-baselines.txt`. The supplied model pickle was not loaded; every configuration was refitted.

## Interpretation

The existing separate-word hybrid achieves test F1 0.868080. Numerical-only and word-text-only alternatives yield 0.825426 and 0.767180. The gains are 4.2653 and 10.0900 percentage points.

The joined-word hybrid yields test F1 0.883802, and the character-3-gram hybrid yields 0.948594. Both also exceed the existing word configuration in each of the five validation runs. The results support the existing block combination but do not show that its word tokenizer or field separation is optimal. The character alternative detects 393 more anomalies and produces 101 fewer false alarms on the seed-42 test set. It has not replaced the original saved training/inspection pipeline, and no model binary is published by this study.

Source-specific current-word-hybrid test results are in `test_by_source.csv`: CSIC n=5,121 (1,927 normal, 3,194 anomalous), F1 0.862919 and FPR 0.137519; local synthetic n=209 (90 normal, 119 anomalous), F1 1.0 and FPR 0.0. Training remains mixed for both rows. All five configurations have source-specific output rows, including the weaker and stronger alternatives.

## Uncertainty and dependence

Validation means and sample SDs describe the five recorded seeds. Seeds change both record splitting and RF randomness; partitions share one corpus. Seed-42 test records can be in training or validation at other seeds. These are not independent datasets or application replications.

Paired test F1-difference intervals use 2,000 label-stratified record bootstraps with a shared resample across both configurations and RNG seed 20261102. They are marginal, unadjusted 95% intervals conditional on this test split. Template dependence and application shift are not represented. No population-level significance or SOTA claim follows from them. `paired_test_f1_intervals.json` contains all four contrasts, including those favoring the baselines.

The archived component study uses score >=0.5; the matched comparison uses estimator predict for every row. Exact RF ties can differ. A separate >=0.5 check reproduces the archived five-seed F1 values for the original hybrid, numerical-only and text-only configurations to floating-point precision. The new original-hybrid test confusion counts also reproduce the supplied primary table: TN=1,752, FP=265, FN=569, TP=2,744.

## Files and checks

- `run_manifest.json`: protocol/script/data/raw-file SHA-256 values, runtime, split-index digests, dimensions and model settings.
- `metrics.csv`: 25 validation rows plus all five seed-42 test rows.
- `validation_summary.csv`: all five mean/SD summaries.
- `predictions/`: row IDs, true labels, scores, decisions and coarse source groups, without duplicated raw payloads.
- `request_origins.csv`: origin audit for every retained row.
- `threshold_ge_half_check.csv`: decision-convention audit, not an alternative table chosen for favorable results.
- `test_by_source.csv`: every configuration evaluated by CSIC/local origin.
- `verification.json`: recomputation and reproduction checks.

All 30 metric rows were recomputed from saved predictions. Use `pd.read_csv(path, float_precision="round_trip")` for floating scores; the default decimal parser can slightly perturb score ties and ROC-AUC. F1, confusion counts and the paired-bootstrap intervals depend on the saved decisions and are unaffected.

Source-specific ROC-AUC is computed by the runner after reading the prediction CSV with its default parser; tiny tie-rounding differences may occur compared with round-trip parsing. The manuscript's source table reports recall, F1 and FPR, which are based on saved labels and decisions.

Fit durations are retained only as diagnostics. They are not a standardized latency or throughput benchmark. No independent-template split, adversarial adaptation, external application trial or recalibration was performed in this study.
