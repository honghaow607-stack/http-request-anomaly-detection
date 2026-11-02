# HTTP Request Anomaly Detection

Hybrid numerical and word-level TF-IDF features for HTTP request anomaly classification, with logistic regression, random forest and XGBoost; command-line and Streamlit inspection reuse a saved fitted pipeline.

The original experiment archive covers a separate validation partition, header-aware deduplication, five-seed component ablations, ATRDF evaluation, cross-subset checks, query re-encoding probes and an interface threshold sweep. The checks in `docs/result_audit.json` audit those supplied outputs. A separately rerun matched representation study, including all five configurations and per-request predictions, is now recorded in `results/representation_baselines/`.

## Quick start

Python 3.10 or newer is recommended. Minimum dependency versions are listed in `requirements.txt`; they are not a lock file for the original training environment.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/prepare_data.py
python run.py
python src/predict.py --text "GET /products?id=1"
streamlit run src/app.py
```

The preparation step downloads **two CSIC files** and copies five fixed local-synthetic files. Do not add the official normal-training file or regenerate examples when reproducing the bundled corpus. No trained pickle is committed: `run.py` saves `model.pkl` locally. Load model files only from a source you trust.

## Representation and protocol

- 40 numerical descriptors: method, lengths/structure, punctuation, encoding/path patterns, entropy and keyword counts.
- Separate target/body word 1–3-gram TF-IDF channels, capped at 3,000/2,000 terms; each uses original and up-to-three-round percent-decoded text.
- 5,040 realized features for the CSIC mixture; 3,440 in the supplied ATRDF ablation. Vocabulary size depends on the training corpus.
- Deduplication key: method, target, canonicalized headers, body and label. Volatile headers are excluded from the key; the classifier retains only a header-length descriptor.
- Stratified **64/16/20** train/validation/test partition; TF-IDF is fitted only on training records. Validation recall selects the classifier, with F1 breaking ties.
- Main classifier-comparison CSVs use estimator `predict`; ablation/ATRDF/probe scripts and interfaces use positive-class score `>= 0.5`. Exact RF ties can therefore differ. Scores are not calibrated probabilities.

The primary corpus contains 62,165 parsed requests and **26,646 retained records**, including local synthetic examples (10,085 normal / 16,561 anomalous). Seed 42 gives 17,052 training, 4,264 validation and 5,330 test records.

## Bundled test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7647 | 0.9228 | 0.6782 | 0.7818 | 0.8461 |
| Random forest | 0.8435 | 0.9119 | 0.8283 | 0.8681 | 0.9274 |
| XGBoost | 0.8098 | 0.9574 | 0.7262 | 0.8260 | 0.9177 |

These are results for the mixed CSIC/local corpus under this protocol, not scores for a CSIC-only test or production traffic. All three classifiers in this table use the same representation. The additional matched encoding comparisons below are controlled adaptations of related representation choices, not complete reproductions of external systems.

## Matched representation baselines

Five fixed representations were fitted with the same 200-tree balanced RF, corpus and seeds (42, 1337, 2024, 7, 99). All use estimator `predict`; exact score-0.5 ties are class 0. Vocabularies are fitted on training records only, with no hyperparameter or threshold search.

| Representation | Validation F1, mean ± sample SD | Seed-42 test recall | Test F1 | Test FPR |
|---|---:|---:|---:|---:|
| Numerical only | 0.8234 ± 0.0043 | 0.8035 | 0.8254 | 0.2355 |
| Word text only | 0.7306 ± 0.0488 | 0.7296 | 0.7672 | 0.2831 |
| Existing separate-word hybrid | 0.8685 ± 0.0052 | 0.8283 | 0.8681 | 0.1314 |
| Joined-word hybrid | 0.8807 ± 0.0046 | 0.8449 | 0.8838 | 0.1101 |
| Character-3-gram hybrid | **0.9488 ± 0.0009** | **0.9469** | **0.9486** | **0.0813** |

The existing combination improves test F1 over numerical-only and word-text-only RF by 4.27 and 10.09 percentage points. The joined-word and character alternatives perform better than the existing word encoding in all five validation runs. Their seed-42 F1 gains over it are 1.57 and 8.05 points. These results support feature-block fusion under this protocol; they do not support superiority of the existing field-separated word encoding. The character configuration is an evaluated alternative; the normal training/inspection command still uses the original word pipeline.

The original word-hybrid test result separates into **5,121 CSIC requests** (F1 0.8629, recall 0.8219, FPR 0.1375) and **209 local synthetic requests** (F1 1.0000, recall 1.0000, FPR 0.0000). Both are evaluated after mixed-corpus training. The synthetic subset is small and is not independent application traffic.

For a reproducible run, first prepare the fixed raw files as above and build the cache with `python -c "from src.data_loader import load_dataset; load_dataset(rebuild=True)"`. Then use the pinned supplementary environment and a new output directory:

```bash
pip install -r docs/requirements-representation-baselines.txt
python scripts/run_representation_baselines.py \
  --protocol results/representation_baselines/protocol.md \
  --output results/representation_baselines_rerun
```

The runner aborts unless the 26,646 cached records exactly match representatives reconstructed from the raw files. It does not silently replace missing data with synthetic examples. See [docs/REPRESENTATION_BASELINES.md](docs/REPRESENTATION_BASELINES.md) for provenance, baseline definitions and conditional uncertainty. Read stored floating scores with pandas `float_precision="round_trip"` when auditing ROC-AUC ties.

## Component ablations

```bash
python scripts/run_ablation.py
python scripts/run_leave_one_out.py
python scripts/run_ablation_multiseed.py
python scripts/run_leave_one_out_atrdf.py
```

Ablations evaluate **validation** records. Five seeds (42, 1337, 2024, 7, 99) each evaluate the full RF plus seven component removals. Removing a feature family zeroes its columns and retrains RF with the same fitted vocabulary; removing a whole representation or changing decoding fits that configuration separately.

| CSIC validation configuration | F1, mean ± sample SD | Difference from full, percentage points |
|---|---:|---:|
| Full hybrid | 0.8688 ± 0.0050 | — |
| Without TF-IDF (numerical only) | 0.8234 ± 0.0039 | -4.53 |
| Without numerical block (text only) | 0.7307 ± 0.0489 | -13.81 |
| One decoding round | 0.8688 ± 0.0051 | +0.005 |

The supplied seed-level values reproduce the summary means, SDs and paired t-test values. For seven F1 comparisons, Bonferroni-adjusted p-values are 0.0000922 (without TF-IDF) and 0.0303 (without numerical features). Other F1 comparisons do not pass that correction. The five runs reuse one corpus with overlapping partitions; these are exploratory checks, not independent population-level replications. Confidence intervals and paired effect sizes are recorded in `docs/ablation_statistics.json`.

ATRDF gives a different contribution pattern: numerical-only RF F1 is 0.9849, compared with 0.9840 for the full hybrid and 0.9532 for text-only. Text features are therefore not uniformly necessary across these corpora. See `ablation/report.md` and the exact CSVs under `ablation/data/`.

## ATRDF and transfer

Get the author-maintained data from [ArielCyber/ATRDF](https://github.com/ArielCyber/Cisco_Ariel_Uni_API_security_challenge); paths and attribution are in [docs/DATA.md](docs/DATA.md).

```bash
python scripts/run_atrdf_eval.py
python scripts/run_atrdf_cross_subset.py
```

The labeled dataset-3 training file is deduplicated to 129,674 records and repartitioned; its test partition contains 25,935 requests. Retrained F1: LR 0.9776, RF 0.9820, XGBoost 0.9851. The CSIC-trained RF transferred without retraining has FPR **0.9917** and F1 0.2224; high recall in that row accompanies rejection of nearly all normal requests.

Dataset-3-trained RF yields F1 0.9358 / 0.9282 / 0.4859 on datasets 1 / 2 / 4. Dataset 4 FPR is 0.6027. Attack-template-key overlap is nonzero (5.46% / 1.54% / 1.26%), so cross-file evaluation must not be described as proven leakage-free. Same-subset test-template overlap is 27.26%. These measurements delimit transfer; high retrained accuracy does not establish cross-domain generalization.

The competition validation files are unlabeled. Their missing Attack_Tag fields do not establish that they are benign, and those files are excluded from labeled evaluation.

## Encoding and interface checks

```bash
python scripts/adversarial_test.py
python scripts/e2e_test.py
```

The encoding probe draws 60 anomalous and 60 normal GET requests from validation/test records and adds zero, one or two rounds of query-value percent encoding. The full hybrid has identical one-/three-round outcomes in this probe. For text-only features at depth 3, three-round decoding reduces FPR from 0.9833 to 0.6000, but this remains high and attack recall changes from 1.0000 to 0.8333. This is a targeted re-encoding probe, not adaptive adversarial robustness.

A submitted CSV column counted 120 total requests as the attack-row sample size. This release corrects that count to 60 and fixes the corresponding script expression; recall/FPR values are unchanged. The 200-request interface threshold sweep samples raw source records and can overlap training data, so it is an implementation check rather than an independent test or latency benchmark.

## Files and reproducibility

- `src/`, `run.py`: parsing, feature construction, classifiers and interfaces.
- `scripts/`: ablation, matched-representation, ATRDF, encoding and interface evaluations.
- `ablation/data/`, `results/`: original aggregate outputs and figures, plus the separately rerun matched study.
- `results/representation_baselines/`: fixed protocol, runtime and input hashes, 30 metric rows, paired intervals, per-request predictions, source breakdowns and verification.
- `data/synthetic/`: fixed local examples used by the preparation command.
- `docs/DATA.md`: third-party sources and expected local paths.
- `docs/result_audit.json`: corpus/cache and confusion-count checks.
- `docs/ablation_statistics.json`: recomputed seed-level F1 contrasts.

The original 1.4 GB extracted corpus/cache and the 111 MB trained model are omitted from regular Git. Code and aggregate evidence remain available; datasets retain their upstream attribution/terms. A source-code license has not been selected for this repository.

## Authors

Honghao Wang, Zihan Ma and Xinyuan Li — Beijing University of Posts and Telecommunications. Correspondence: wanghonghao@bupt.edu.cn.
