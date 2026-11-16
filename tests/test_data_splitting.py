import pandas as pd
import pytest

from data_splitting.split import split_by_pcap


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
    train, val, test = result
    assign = pd.Series("", index=meta.index)
    assign.iloc[train] = "train"
    assign.iloc[val] = "val"
    assign.iloc[test] = "test"
    return assign


def test_chronological_assignment():
    meta = _flow_meta(apps=("A",), pcaps_per_app=10, flows_per_pcap=1)
    train, val, test = split_by_pcap(meta)

    train_pcaps = sorted(meta.iloc[train]["file_source"])
    assert train_pcaps == [f"Qom-A-{i}.pcap" for i in range(0, 7)]
    assert sorted(meta.iloc[val]["file_source"]) == ["Qom-A-7.pcap"]
    assert sorted(meta.iloc[test]["file_source"]) == [
        f"Qom-A-{i}.pcap" for i in range(8, 10)
    ]


def test_pcap_flows_stay_together():
    meta = _flow_meta(apps=("A", "B"), pcaps_per_app=10, flows_per_pcap=3)
    assign = _assignment(meta, split_by_pcap(meta))
    per_pcap_unique = assign.groupby(meta["file_source"]).nunique()
    assert (per_pcap_unique == 1).all()
    assert not (assign == "").any()


def test_each_set_contains_all_apps():
    meta = _flow_meta(apps=("A", "B", "C"), pcaps_per_app=10, flows_per_pcap=2)
    train, val, test = split_by_pcap(meta)
    assert set(meta.iloc[train]["application"]) == {"A", "B", "C"}
    assert set(meta.iloc[val]["application"]) == {"A", "B", "C"}
    assert set(meta.iloc[test]["application"]) == {"A", "B", "C"}


def test_custom_ratios():
    meta = _flow_meta(apps=("A",), pcaps_per_app=10, flows_per_pcap=1)
    train, val, test = split_by_pcap(meta, ratios=(0.5, 0.25, 0.25))
    assert len(train) == 5
    assert len(val) == 2
    assert len(test) == 3


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
    train, val, test = split_by_pcap(meta, dataset_scope="per_dataset")

    qom_train = {
        meta.iloc[i]["file_source"]
        for i in train
        if meta.iloc[i]["dataset_name"] == "Qom"
    }
    iraq_train = {
        meta.iloc[i]["file_source"]
        for i in train
        if meta.iloc[i]["dataset_name"] == "Iraq"
    }
    assert qom_train == {"Qom-0.pcap", "Qom-1.pcap"}
    assert iraq_train == {"Iraq-0.pcap", "Iraq-1.pcap"}


def test_per_dataset_scope_requires_dataset_column():
    meta = _flow_meta(apps=("A",), pcaps_per_app=4, flows_per_pcap=1).drop(
        columns="dataset_name"
    )
    with pytest.raises(ValueError, match="dataset_name"):
        split_by_pcap(meta, dataset_scope="per_dataset")


def test_invalid_ratios():
    meta = _flow_meta(apps=("A",), pcaps_per_app=4, flows_per_pcap=1)
    with pytest.raises(ValueError):
        split_by_pcap(meta, ratios=(0.6, 0.2, 0.1))  # does not sum to 1
    with pytest.raises(ValueError):
        split_by_pcap(meta, ratios=(0.5, 0.5))  # wrong length


def test_invalid_scope():
    with pytest.raises(ValueError):
        split_by_pcap(_flow_meta(), dataset_scope="nope")
