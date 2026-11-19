"""Train and evaluate classical classifiers on extracted flow features.

Usage: ``python train.py`` (composes ``config/train.yaml``) or override any
field, e.g. ``python train.py classification.models.0.enabled=false`` or
``python train.py data_splitting.n_folds=3``.

Requires the flow-feature-extraction and data-splitting modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

from data_splitting.filter import cap_pcaps_per_app, filter_applications
from data_splitting.split import cross_validate_by_pcap, split_by_pcap
from evaluation.metrics import (
    compute_metrics,
    confusion_matrix_dataframe,
    full_report,
    report_to_dataframe,
)
from models.classical.classifiers import build_classifier
from preprocessing.feature_extraction import make_extractor, prepare_packet_dataframe
from utils.schema.train import TrainConfig


def _save_model_reports(
    name: str, y_true, y_pred, zero_division: int, models_dir: Path
) -> None:
    out_dir = models_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_to_dataframe(
        full_report(y_true, y_pred, zero_division=zero_division)
    ).to_csv(out_dir / "classification_report.csv", index=False)
    confusion_matrix_dataframe(y_true, y_pred).to_csv(out_dir / "confusion_matrix.csv")


def _run_holdout(X, y, y_enc, le, train_idx, test_idx, cls) -> list[dict]:
    models_dir = Path(cls.models_dir)
    rows: list[dict] = []
    for spec in cls.models:
        if not spec.enabled:
            continue
        clf = build_classifier(spec.name, spec.params)
        clf.fit(X.iloc[train_idx], y_enc[train_idx])

        pred_train = le.inverse_transform(clf.predict(X.iloc[train_idx]))
        pred_test = le.inverse_transform(clf.predict(X.iloc[test_idx]))
        for split_name, idx, pred in (
            ("train", train_idx, pred_train),
            ("test", test_idx, pred_test),
        ):
            m = compute_metrics(y.iloc[idx], pred, zero_division=cls.zero_division)
            rows.append({"model": spec.name, "split": split_name, **m})

        _save_model_reports(
            spec.name, y.iloc[test_idx], pred_test, cls.zero_division, models_dir
        )
    return rows


def _run_cv(X, y, y_enc, le, folds, cls) -> list[dict]:
    models_dir = Path(cls.models_dir)
    rows: list[dict] = []
    for spec in cls.models:
        if not spec.enabled:
            continue
        clf = build_classifier(spec.name, spec.params)

        per_fold: list[dict] = []
        y_true_pool: list[np.ndarray] = []
        y_pred_pool: list[np.ndarray] = []
        for fold, (train_idx, test_idx) in enumerate(folds):
            clf.fit(X.iloc[train_idx], y_enc[train_idx])
            pred = le.inverse_transform(clf.predict(X.iloc[test_idx]))
            m = compute_metrics(y.iloc[test_idx], pred, zero_division=cls.zero_division)
            per_fold.append(m)
            rows.append({"model": spec.name, "fold": fold, **m})
            y_true_pool.append(y.iloc[test_idx].to_numpy())
            y_pred_pool.append(np.asarray(pred))

        agg = pd.DataFrame(per_fold)
        means = agg.mean()
        stds = agg.std()
        rows.append(
            {"model": spec.name, "fold": "mean", **{k: means[k] for k in means.index}}
        )
        rows.append(
            {"model": spec.name, "fold": "std", **{k: stds[k] for k in stds.index}}
        )

        _save_model_reports(
            spec.name,
            np.concatenate(y_true_pool),
            np.concatenate(y_pred_pool),
            cls.zero_division,
            models_dir,
        )
    return rows


def run_classification(config: TrainConfig) -> pd.DataFrame:
    """Extract features, split by pcap, and train/evaluate each enabled model."""
    cls = config.classification
    ds = config.data_splitting
    prep = config.preprocessing

    pkt_df = pd.read_csv(cls.combined_csv)
    pkt_df = filter_applications(pkt_df, cls.applications)
    pkt_df = cap_pcaps_per_app(pkt_df, cls.max_pcaps_per_app)
    pkt_df = prepare_packet_dataframe(pkt_df)

    extractor = make_extractor(
        feature_config=prep.feature_config,
        rate_interval=prep.rate_interval,
        interactive_bins=prep.interactive_bins,
    )
    X = extractor.transform(pkt_df)
    flow_meta = extractor.flow_meta
    y = flow_meta["application"]

    # Encode string labels to numeric for estimators that require them
    # (e.g. XGBoost); predictions are decoded back before reporting.
    le = LabelEncoder()
    y_enc = le.fit_transform(y.to_numpy())

    if ds.n_folds is not None:
        folds = cross_validate_by_pcap(flow_meta, ds.n_folds, ds.dataset_scope, ds.seed)
        rows = _run_cv(X, y, y_enc, le, folds, cls)
    else:
        train_idx, test_idx = split_by_pcap(flow_meta, ds.test_size, ds.dataset_scope)
        rows = _run_holdout(X, y, y_enc, le, train_idx, test_idx, cls)

    out = pd.DataFrame(rows)
    out.to_csv(cls.metrics_csv, index=False)
    return out


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    schema = OmegaConf.structured(TrainConfig)
    user_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != "hydra"})
    config = OmegaConf.to_object(OmegaConf.merge(schema, user_cfg))
    metrics = run_classification(config)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
