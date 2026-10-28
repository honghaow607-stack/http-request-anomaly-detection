# Reproducibility notes

## Reference inputs and results

`data_manifest.json` records original-byte SHA256 values for the seven input files from the original project. The reference experiment parsed 62,165 requests and retained 26,572 after removing empty targets and deduplicating by method, target, body and label. It used two CSIC files, not the official normal-training file, together with five synthetic files. Labels follow source files; the detector itself is binary.

`tools/prepare_data.py --mode paper` copies the bundled synthetic inputs and downloads only the two specified CSIC files. Mirrors are pinned to commit `424c6e1c1951570473d9d6c0227eaaac6fdd6d02` of `msudol/Web-Application-Attack-Datasets`. Those mirror copies match the original request contents after CRLF-to-LF newline normalization; their raw-byte hashes can differ from the Windows originals. For CSIC inputs the tool accepts either the original-byte hash or the separately recorded LF-normalized hash. Other content changes are rejected. The original parser uses universal newlines, so this does not alter the parsed requests. The command rebuilds `data/processed/dataset.csv`. When a mirror is unavailable or changes, it fails explicitly; `--source-dir` accepts the original local CSIC files instead. The tool does not remove unexpected raw files or silently replace them.

`--mode synthetic` uses only the bundled synthetic inputs: 1,100 raw requests and 986 retained requests. It is an offline demonstration protocol. Its scores cannot be substituted for the reference results. Switching protocols requires a clean working copy or backing up and moving the previous raw and processed directories.

## Training protocol

The original `run.py` performs a stratified 80/20 split with random state 42. `FeaturePipeline.fit_transform` fits target/body vocabularies on training requests; evaluation calls `transform`. All three estimators receive the same concatenated representation. The reference corpus reaches both vocabulary limits, yielding 40 + 1,000 + 500 features. Numerical features are not scaled separately.

The estimator definitions remain in `src/models.py`. Unspecified parameters use library defaults. The selected model maximizes evaluation recall, with F1 as the tie-breaker. The selected model and fitted pipeline are serialized together. Interactive prediction uses the positive-class score at threshold 0.5.

## Reference artifacts

`results/reference/` contains the original metric CSV and seven supplied PNG plots. They were copied from the author's project without replacing the scores. FPR values in README are derived from the supplied confusion matrices:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic regression | 1,802 | 205 | 1,118 | 2,190 |
| Random forest | 1,660 | 347 | 531 | 2,777 |
| XGBoost | 1,870 | 137 | 917 | 2,391 |

These counts reproduce Accuracy, Precision, Recall and F1 to the displayed precision. Reconstructing ROC-AUC independently requires per-request scores; those are not present in the original summary CSV. Fresh runs write to `results/`, leaving `results/reference/` unchanged.

## Environment and provenance

`requirements.txt` specifies minimum dependencies, not a recovered lockfile for the author's experiment. `scipy` is listed explicitly because the feature pipeline imports it directly. Exact historical training hardware and package versions have not been recovered. A successful local check of the publishing package must not be represented as independent reproduction of the author's classifier scores.

`source_manifest.json` preserves SHA256 for the imported `run.py` and `src/*.py` implementation files. The publishing work adds README, examples, a data-preparation wrapper and metadata. It does not alter the original parsing, learning algorithm or selection rule. `.gitattributes` preserves the bytes of implementation and synthetic data files across checkouts.

For a new experiment, retain the data manifest, commit hash, installed package versions, hardware, split row IDs and per-request `y_true`, `y_pred`, `score`, `source_file` and threshold. Do not claim matching scores merely because the request count agrees.

## Limits to interpretation

The reported evaluation partition is also used to select the saved model. No independent final test, repeated-seed estimates, feature ablation, augmentation attribution, calibration, cross-site validation or deployment profiling is supplied. Deduplication of raw requests does not establish uniqueness of model-visible feature vectors. Keyword matches and feature-importance plots do not establish faithful local explanations.

The percentage-decoding step uses `unquote`, retains the original text, and performs at most three passes. It does not apply form-style plus-to-space decoding or guarantee agreement with an application's own parser. The prototype reads HTTP request text after it is available to the user; it does not intercept or decrypt HTTPS connections.
