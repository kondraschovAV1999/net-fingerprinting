import pandas as pd
import pytest

from data_splitting.split import (
    cross_validate_by_pcap,
    min_pcaps_per_app,
    split_by_pcap,
)


def _flow_meta(apps=("A", "B"), pcaps_per_app=10, flows_per_pcap=2, datasets=("Qom",)):
    rows = []
    for ds in datasets:
        for app in apps:
            for i in range(pcaps_per_app):
                fsrc = f"{ds}-{app}-{i}.pcap"
                ts = 1000 + i
                for _ in range(flows_per_pcap):
                    rows.append(
                        {
                            "file_source": fsrc,
                            "application": app,
                            "capture_timestamp": ts,
                            "dataset_name": ds,
                        }
                    )
    return pd.DataFrame(rows)


def _assignment(meta, result):
    train, test = result
    assign = pd.Series("", index=meta.index)
    assign.iloc[train] = "train"
    assign.iloc[test] = "test"
    return assign


def test_chronological_assignment():
    meta = _flow_meta(apps=("A",), pcaps_per_app=10, flows_per_pcap=1)
    train, test = split_by_pcap(meta, test_size=0.15)

    # n_test = ceil(0.15 * 10) = 2 newest pcaps -> test
    assert sorted(meta.iloc[test]["file_source"]) == [
        "Qom-A-8.pcap",
        "Qom-A-9.pcap",
    ]
    assert sorted(meta.iloc[train]["file_source"]) == [
        f"Qom-A-{i}.pcap" for i in range(0, 8)
    ]


def test_pcap_flows_stay_together():
    meta = _flow_meta(apps=("A", "B"), pcaps_per_app=10, flows_per_pcap=3)
    assign = _assignment(meta, split_by_pcap(meta))
    per_pcap_unique = assign.groupby(meta["file_source"]).nunique()
    assert (per_pcap_unique == 1).all()
    assert not (assign == "").any()


def test_each_set_contains_all_apps():
    meta = _flow_meta(apps=("A", "B", "C"), pcaps_per_app=10, flows_per_pcap=2)
    train, test = split_by_pcap(meta)
    assert set(meta.iloc[train]["application"]) == {"A", "B", "C"}
    assert set(meta.iloc[test]["application"]) == {"A", "B", "C"}


def test_small_app_no_crash():
    meta = _flow_meta(apps=("A",), pcaps_per_app=3, flows_per_pcap=1)
    train, test = split_by_pcap(meta)
    assert len(test) == 1
    assert len(train) == 2


def test_single_pcap_app_dropped():
    meta = _flow_meta(apps=("A",), pcaps_per_app=1, flows_per_pcap=1)
    train, test = split_by_pcap(meta)
    assert len(train) == 0
    assert len(test) == 0


def test_per_dataset_scope():
    rows = []
    for ds in ("Qom", "Iraq"):
        for i in range(4):
            rows.append(
                {
                    "file_source": f"{ds}-{i}.pcap",
                    "application": "A",
                    "capture_timestamp": 1000 + i,
                    "dataset_name": ds,
                }
            )
    meta = pd.DataFrame(rows)
    train, test = split_by_pcap(meta, dataset_scope="per_dataset")

    qom_test = {
        meta.iloc[i]["file_source"]
        for i in test
        if meta.iloc[i]["dataset_name"] == "Qom"
    }
    assert qom_test == {"Qom-3.pcap"}


def test_per_dataset_scope_requires_dataset_column():
    meta = _flow_meta(apps=("A",), pcaps_per_app=4, flows_per_pcap=1).drop(
        columns="dataset_name"
    )
    with pytest.raises(ValueError, match="dataset_name"):
        split_by_pcap(meta, dataset_scope="per_dataset")


def test_invalid_test_size():
    meta = _flow_meta(apps=("A",), pcaps_per_app=4, flows_per_pcap=1)
    with pytest.raises(ValueError):
        split_by_pcap(meta, test_size=1.5)
    with pytest.raises(ValueError):
        split_by_pcap(meta, test_size=0.0)


def test_invalid_scope():
    with pytest.raises(ValueError):
        split_by_pcap(_flow_meta(), dataset_scope="nope")


def _meta_varying(apps_pcaps, flows_per_pcap=2):
    rows = []
    for app, n in apps_pcaps.items():
        for i in range(n):
            fsrc = f"{app}-{i}.pcap"
            for _ in range(flows_per_pcap):
                rows.append(
                    {
                        "file_source": fsrc,
                        "application": app,
                        "capture_timestamp": 1000 + i,
                        "dataset_name": "Qom",
                    }
                )
    return pd.DataFrame(rows)


def test_min_pcaps_per_app_combined():
    meta = _meta_varying({"A": 3, "B": 6})
    assert min_pcaps_per_app(meta) == 3


def test_cross_validate_fold_count():
    meta = _meta_varying({"A": 3, "B": 6})
    assert len(cross_validate_by_pcap(meta, n_folds=3)) == 3


def test_cross_validate_over_cap_raises():
    meta = _meta_varying({"A": 3, "B": 6})
    with pytest.raises(ValueError):
        cross_validate_by_pcap(meta, n_folds=4)


def test_cross_validate_pcap_group_invariant():
    meta = _meta_varying({"A": 3, "B": 6}, flows_per_pcap=3)
    for train, test in cross_validate_by_pcap(meta, n_folds=3):
        train_pcaps = set(meta.iloc[train]["file_source"])
        test_pcaps = set(meta.iloc[test]["file_source"])
        assert train_pcaps.isdisjoint(test_pcaps)


def test_cross_validate_each_pcap_in_exactly_one_test_fold():
    meta = _meta_varying({"A": 3, "B": 6}, flows_per_pcap=1)
    seen = {}
    for fold, (_train, test) in enumerate(cross_validate_by_pcap(meta, n_folds=3)):
        for idx in test:
            fsrc = meta.iloc[idx]["file_source"]
            assert fsrc not in seen
            seen[fsrc] = fold
    assert set(seen) == set(meta["file_source"])


def test_cross_validate_invalid_n_folds():
    meta = _meta_varying({"A": 3, "B": 6})
    with pytest.raises(ValueError):
        cross_validate_by_pcap(meta, n_folds=1)
    with pytest.raises(ValueError):
        cross_validate_by_pcap(meta, n_folds=2.5)
