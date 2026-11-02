"""Run fixed, matched representation comparisons without tuning on test data.

The joined-word and character-3-gram alternatives are controlled adaptations
of representation choices in related work, not full reproductions of papers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import _canonical_headers, parse_request_file
from src.features import FeaturePipeline, TFIDF_TOKEN_PATTERN
from src.preprocess import preprocess_dataset

SEEDS = (42, 1337, 2024, 7, 99)
CONFIGURATIONS = (
    "hybrid_word_separate",
    "numerical_only",
    "word_text_only",
    "hybrid_word_joined",
    "hybrid_char3_separate",
)
TEXT_COLUMNS = ("method", "url", "path", "query", "headers", "body")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def reconstruct_origins(data: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Rebuild representatives and require exact agreement with the cache."""
    frames = []
    for path in sorted(raw_dir.glob("*.txt")):
        name = path.stem.lower()
        if name.startswith("synthetic_"):
            attack_type = name.removeprefix("synthetic_")
            label = int(attack_type != "normal")
            source_group = "local_synthetic"
        elif any(token in name for token in ("anomal", "attack", "malicious")):
            label, attack_type, source_group = 1, "anomalous", "csic"
        elif "normal" in name:
            label, attack_type, source_group = 0, "normal", "csic"
        else:
            continue
        frame = pd.DataFrame(parse_request_file(path, label, attack_type))
        frame["source_file"] = path.name
        frame["source_group"] = source_group
        frames.append(frame)
    if not frames:
        raise ValueError("No recognized raw source files; origin audit cannot proceed")
    rebuilt = pd.concat(frames, ignore_index=True)
    rebuilt = rebuilt.loc[rebuilt["url"].str.len() > 0].copy()
    rebuilt["_canonical_headers"] = rebuilt["headers"].map(_canonical_headers)
    rebuilt = rebuilt.drop_duplicates(
        ["method", "url", "_canonical_headers", "body", "label"]
    ).reset_index(drop=True)
    if not rebuilt[data.columns].equals(data):
        raise ValueError("Raw representatives do not exactly match the supplied cache")
    return rebuilt[["source_file", "source_group"]]


def evaluate(y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "records": int(len(y_true)),
        "normal": int(np.count_nonzero(y_true == 0)),
        "anomalous": int(np.count_nonzero(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else None,
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "exact_half_scores": int(np.count_nonzero(score == 0.5)),
    }


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=5000, ngram_range=(1, 3),
        token_pattern=TFIDF_TOKEN_PATTERN, sublinear_tf=True,
    )


def alternative_matrices(
    name: str,
    prepared: dict[str, pd.DataFrame],
    full: dict[str, sparse.csr_matrix],
) -> dict[str, sparse.csr_matrix]:
    if name == "hybrid_word_separate":
        return full
    if name == "numerical_only":
        return {key: value[:, :40].tocsr() for key, value in full.items()}
    if name == "word_text_only":
        return {key: value[:, 40:].tocsr() for key, value in full.items()}
    if name == "hybrid_word_joined":
        documents = {
            key: frame["text_url"] + " " + frame["text_body"]
            for key, frame in prepared.items()
        }
        vectorizer = word_vectorizer()
        text = {"train": vectorizer.fit_transform(documents["train"])}
        text.update({
            key: vectorizer.transform(value)
            for key, value in documents.items() if key != "train"
        })
        return {
            key: sparse.hstack([full[key][:, :40], value], format="csr")
            for key, value in text.items()
        }
    if name == "hybrid_char3_separate":
        text_blocks: dict[str, list] = {key: [] for key in full}
        for column, maximum in (("text_url", 3000), ("text_body", 2000)):
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(3, 3), max_features=maximum,
                lowercase=True, sublinear_tf=True,
            )
            text_blocks["train"].append(vectorizer.fit_transform(prepared["train"][column]))
            for key in full:
                if key != "train":
                    text_blocks[key].append(vectorizer.transform(prepared[key][column]))
        return {
            key: sparse.hstack([full[key][:, :40], *blocks], format="csr")
            for key, blocks in text_blocks.items()
        }
    raise ValueError(f"Unknown configuration: {name}")


def paired_test_intervals(output: Path) -> list[dict]:
    """Stratified paired bootstrap, conditional on the one observed test split."""
    frames = {
        name: pd.read_csv(output / f"predictions/42_{name}_test.csv").sort_values("row_id")
        for name in CONFIGURATIONS
    }
    reference = frames["hybrid_word_separate"]
    y = reference["label"].to_numpy()
    normal_ids = np.flatnonzero(y == 0)
    anomalous_ids = np.flatnonzero(y == 1)
    rng = np.random.default_rng(20261102)
    deltas = {name: [] for name in CONFIGURATIONS if name != "hybrid_word_separate"}
    reference_prediction = reference["prediction"].to_numpy()
    for frame in frames.values():
        if not np.array_equal(frame["row_id"], reference["row_id"]):
            raise ValueError("Paired test prediction rows differ")
        if not np.array_equal(frame["label"], y):
            raise ValueError("Paired test labels differ")
    for _ in range(20):
        normals = rng.choice(normal_ids, size=(100, len(normal_ids)), replace=True)
        anomalies = rng.choice(anomalous_ids, size=(100, len(anomalous_ids)), replace=True)

        def batch_f1(prediction: np.ndarray) -> np.ndarray:
            tp = prediction[anomalies].sum(axis=1)
            fp = prediction[normals].sum(axis=1)
            fn = len(anomalous_ids) - tp
            return 2 * tp / (2 * tp + fp + fn)

        reference_f1 = batch_f1(reference_prediction)
        for name in deltas:
            prediction = frames[name]["prediction"].to_numpy()
            deltas[name].extend((reference_f1 - batch_f1(prediction)).tolist())
    result = []
    reference_f1 = f1_score(y, reference_prediction)
    for name, distribution in deltas.items():
        low, high = np.quantile(distribution, [0.025, 0.975])
        result.append({
            "baseline": name, "reference": "hybrid_word_separate",
            "delta_f1_pp": float(100 * (reference_f1 - f1_score(y, frames[name]["prediction"]))),
            "conditional_ci95_pp": [float(100 * low), float(100 * high)],
            "bootstrap_resamples": 2000, "stratified_by_label": True,
            "scope": "Conditional record-level test resampling; not independent-template inference",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/dataset.csv")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/representation_baselines")
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "run_manifest.json").exists():
        raise FileExistsError("Output already contains a run; choose a new output directory")
    data = pd.read_csv(args.data, encoding="utf-8-sig").fillna("")
    if len(data) != 26646 or data["label"].value_counts().to_dict() != {1: 16561, 0: 10085}:
        raise ValueError("The fixed protocol requires the supplied 26,646-row corpus")
    origins = reconstruct_origins(data, args.raw_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)
    shutil.copyfile(args.protocol, args.output / "protocol.md")
    origins.assign(row_id=data.index).to_csv(args.output / "request_origins.csv", index=False)
    manifest = {
        "status": "running", "seeds": SEEDS, "configurations": CONFIGURATIONS,
        "data_sha256": sha256(args.data), "protocol_sha256": sha256(args.protocol),
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in (
            PROJECT_ROOT / "src/features.py", PROJECT_ROOT / "src/preprocess.py",
            PROJECT_ROOT / "src/data_loader.py",
        )},
        "raw_file_sha256": {p.name: sha256(p) for p in sorted(args.raw_dir.glob("*.txt"))},
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "platform": platform.platform(),
        },
        "decision_rule": "estimator.predict; exact score-0.5 ties assigned class 0",
        "raw_cache_exact_match": True,
        "origin_counts": origins["source_group"].value_counts().to_dict(),
        "splits": [],
    }
    save_json(args.output / "run_manifest.json", manifest)
    print("Protocol and input hashes recorded; all five configurations will be reported.", flush=True)
    rows: list[dict] = []
    threshold_rows: list[dict] = []
    for seed in SEEDS:
        train, test = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=seed)
        train, validation = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=seed)
        partitions = {"train": train, "validation": validation}
        if seed == 42:
            partitions["test"] = test
        if set(train.index) & (set(validation.index) | set(test.index)) or set(validation.index) & set(test.index):
            raise AssertionError("Partitions overlap within one seed")
        manifest["splits"].append({
            "seed": seed,
            "counts": {key: len(frame) for key, frame in (("train", train), ("validation", validation), ("test", test))},
            "index_sha256": {
                key: hashlib.sha256(frame.index.to_numpy(dtype="<i8").tobytes()).hexdigest()
                for key, frame in (("train", train), ("validation", validation), ("test", test))
            },
        })
        prepared = {key: preprocess_dataset(frame, max_rounds=3) for key, frame in partitions.items()}
        pipeline = FeaturePipeline()
        full = {"train": pipeline.fit_transform(train)}
        if len(pipeline.rule_feature_names_) != 40:
            raise AssertionError("Numerical feature count changed")
        full.update({key: pipeline.transform(frame) for key, frame in partitions.items() if key != "train"})
        for name in CONFIGURATIONS:
            matrices = alternative_matrices(name, prepared, full)
            model = RandomForestClassifier(
                n_estimators=200, criterion="gini", max_depth=None,
                min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                bootstrap=True, class_weight="balanced", random_state=seed, n_jobs=-1,
            )
            started = time.perf_counter()
            print(f"seed={seed} {name}: fit {matrices['train'].shape}", flush=True)
            model.fit(matrices["train"], train["label"].to_numpy())
            fit_seconds = time.perf_counter() - started
            for split, frame in partitions.items():
                if split == "train":
                    continue
                y_true = frame["label"].to_numpy()
                prediction = model.predict(matrices[split])
                score = model.predict_proba(matrices[split])[:, 1]
                row = {"seed": seed, "configuration": name, "split": split,
                       "features": int(matrices[split].shape[1]),
                       "fit_seconds_diagnostic_only": fit_seconds,
                       **evaluate(y_true, prediction, score)}
                rows.append(row)
                threshold_rows.append({"seed": seed, "configuration": name, "split": split,
                                       **evaluate(y_true, (score >= 0.5).astype(int), score)})
                predictions = pd.DataFrame({
                    "row_id": frame.index, "label": y_true, "prediction": prediction,
                    "score": score, "source_group": origins.loc[frame.index, "source_group"].to_numpy(),
                })
                predictions.to_csv(args.output / f"predictions/{seed}_{name}_{split}.csv", index=False)
                print(f"  {split}: F1={row['f1']:.6f} recall={row['recall']:.6f} FPR={row['fpr']:.6f}", flush=True)
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
            pd.DataFrame(threshold_rows).to_csv(args.output / "threshold_ge_half_check.csv", index=False)
            del model, matrices
        save_json(args.output / "run_manifest.json", manifest)
    metrics = pd.DataFrame(rows)
    validation = metrics.loc[metrics["split"] == "validation"]
    summary = validation.groupby("configuration", sort=False)[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(args.output / "validation_summary.csv")
    source_rows = []
    for name in CONFIGURATIONS:
        predictions = pd.read_csv(args.output / f"predictions/42_{name}_test.csv")
        for origin, frame in predictions.groupby("source_group"):
            source_rows.append({"configuration": name, "source_group": origin,
                                **evaluate(frame["label"].to_numpy(), frame["prediction"].to_numpy(), frame["score"].to_numpy())})
    pd.DataFrame(source_rows).to_csv(args.output / "test_by_source.csv", index=False)
    save_json(args.output / "paired_test_f1_intervals.json", paired_test_intervals(args.output))
    manifest["status"] = "complete"
    manifest["model_parameters"] = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ).get_params()
    save_json(args.output / "run_manifest.json", manifest)
    print("Complete: all prespecified configurations, per-request predictions and source breakdowns saved.", flush=True)


if __name__ == "__main__":
    main()
