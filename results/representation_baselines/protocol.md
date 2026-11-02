# Fixed supplementary comparison protocol

Recorded before any baseline fitting in this revision. This is a local prospective protocol, not an external preregistration.

Question: on the supplied CSIC/local mixture, what does the existing representation gain relative to numerical-only and text-only features, a joined word vocabulary, and a field-separated character-3-gram alternative?

Input: the existing 26,646-row cached corpus, checked against its seven raw source files. No new synthetic data, downsampling, or synthetic fallback. Use the same 64/16/20 stratified record split and seed list [42, 1337, 2024, 7, 99] as the archived component study. Training vocabularies fit on training records only. Each seed defines its own split; repeated seeds are not independent corpora.

Prespecified configurations, all retained regardless of ranking:
1. Existing hybrid: 40 numerical columns plus separate target/body word 1–3-gram TF-IDF with caps 3,000/2,000, original plus up-to-three-round decoded text, existing custom token pattern and sublinear TF.
2. Numerical-only RF: the same 40 numerical columns.
3. Text-only RF: the same two word TF-IDF blocks, without numerical columns.
4. Joined-word hybrid RF: the same 40 numerical columns and text content, but a single word 1–3-gram TF-IDF vocabulary capped at 5,000. Same token pattern and sublinear TF. This isolates joined versus field-separated vectorization under a common feature budget.
5. Character-3-gram hybrid RF: the same 40 numerical columns and original/decoded text; separate target/body character 3-gram TF-IDF capped at 3,000/2,000. Use analyzer=char, ngram_range=(3,3), lowercase=True, sublinear_tf=True, the same TF-IDF defaults.

Models: RandomForestClassifier(n_estimators=200, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=balanced, random_state=seed, n_jobs=-1). No hyperparameter or decision-threshold search. Decision rule: estimator predict, so exact score 0.5 ties are class 0. Record >=0.5 metrics separately as a convention check, not for choosing a flattering row.

Report validation results for all five seeds and seed-42 test results for every configuration. Primary endpoint F1; additionally precision, recall, accuracy, ROC-AUC, FPR and confusion counts. Record source-separated CSIC/local test metrics after exact origin reconstruction, without inventing subgroup counts. Save row IDs, true labels, scores and decisions (no need to duplicate raw payloads).

Uncertainty: mean ± sample SD over validation seeds. Descriptive paired seed deltas. Conditional stratified paired-bootstrap intervals for seed-42 test F1 differences (2,000 draws, RNG seed 20261102), retaining the same request resample for both models. These are conditional record-level intervals, not evidence on independent templates or applications. No formal SOTA or population-significance claim.

Literature relation: joined-word comparison motivated by Song et al. (10.9708/jksci.2026.31.01.109); character representation motivated by Shaik et al. (10.5539/cis.v19n2p124). These are controlled adaptations, not exact replications of their full inputs, tuning or published protocols. Do not copy their published scores into this matched table.

Environment: use the available Python 3.12.14, scikit-learn 1.3.2, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.11.4. The supplied model contains a scikit-learn 1.3.0 metadata marker and will not be loaded. Refit the existing configuration as well as baselines in the same environment; report any difference from archived results.
