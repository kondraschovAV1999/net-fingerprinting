"""Train and evaluate classical classifiers on extracted flow features.

Usage: ``python train.py`` (defaults from ``config/classification/default.yaml``)
or override any field, e.g. ``python train.py models.0.enabled=false``.

Requires the flow-feature-extraction and data-splitting modules.
"""

from __future__ import annotations

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from data_splitting.split import split_by_pcap
from evaluation.metrics import compute_metrics
from models.classical.classifiers import build_classifier
from preprocessing.feature_extraction import make_extractor, prepare_packet_dataframe
from utils.schema.classification import ClassificationConfig


def run_classification(config: ClassificationConfig) -> pd.DataFrame:
    """Extract features, split by pcap, and train/evaluate each enabled model."""
    pkt_df = pd.read_csv(config.combined_csv)
    pkt_df = prepare_packet_dataframe(pkt_df)

    extractor = make_extractor(
        feature_config=config.feature_config,
        rate_interval=config.rate_interval,
        interactive_bins=config.interactive_bins,
    )
    X = extractor.transform(pkt_df)
    flow_meta = extractor.flow_meta
    y = flow_meta["application"]

    train_idx, val_idx, test_idx = split_by_pcap(
        flow_meta, ratios=config.ratios, dataset_scope=config.dataset_scope
    )

    rows: list[dict] = []
    for spec in config.models:
        if not spec.enabled:
            continue
        clf = build_classifier(spec.name, spec.params)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        for split_name, idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            pred = clf.predict(X.iloc[idx])
            metrics = compute_metrics(
                y.iloc[idx], pred, zero_division=config.zero_division
            )
            rows.append({"model": spec.name, "split": split_name, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(config.metrics_csv, index=False)
    return out


@hydra.main(
    version_base=None, config_path="config/classification", config_name="default"
)
def main(cfg: DictConfig) -> None:
    schema = OmegaConf.structured(ClassificationConfig)
    user_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != "hydra"})
    config = OmegaConf.to_object(OmegaConf.merge(schema, user_cfg))
    metrics = run_classification(config)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
