import pandas as pd
import pytest

from preprocessing.split import assign_pcap_splits, chronological_pcap_split


def _pcaps(app, n, start=1_700_000_000):
    """n pcaps of one app, deliberately listed newest first."""
    return pd.DataFrame({
        "application": app,
        "file_source": [f"{start + i}_{app}_1.0-1.0.pcap" for i in reversed(range(n))],
        "capture_timestamp": [start + i for i in reversed(range(n))],
    })


@pytest.mark.parametrize("n, expected", [
    (3, (1, 1, 1)),
    (4, (2, 1, 1)),   # Qom Google Drive
    (6, (4, 1, 1)),   # Qom Duolingo
    (7, (5, 1, 1)),   # Iraq Duolingo / LinkedIn
    (8, (6, 1, 1)),   # Qom LinkedIn
    (12, (8, 2, 2)),  # Iraq Google Drive
])
def test_split_sizes_per_app(n, expected):
    counts = assign_pcap_splits(_pcaps("App", n))["split"].value_counts()
    assert (counts.get("train", 0), counts.get("val", 0), counts.get("test", 0)) == expected


def test_oldest_pcaps_train_newest_test():
    pcaps = assign_pcap_splits(_pcaps("App", 7))
    assert pcaps["capture_timestamp"].is_monotonic_increasing
    assert pcaps["split"].tolist() == ["train"] * 5 + ["val", "test"]


def test_each_app_is_split_separately():
    pcaps = assign_pcap_splits(pd.concat([_pcaps("A", 3), _pcaps("B", 8, start=1_600_000_000)]))
    for app in ("A", "B"):
        assert set(pcaps.loc[pcaps["application"] == app, "split"]) == {"train", "val", "test"}


def test_flows_follow_their_pcap():
    pcaps = _pcaps("App", 3)
    flow_info = pd.DataFrame({
        "application": "App",
        "file_source": [pcaps["file_source"][i] for i in (0, 0, 1, 2, 2)],
        "capture_timestamp": [pcaps["capture_timestamp"][i] for i in (0, 0, 1, 2, 2)],
    }, index=[f"flow{i}" for i in range(5)])
    split = chronological_pcap_split(flow_info)
    assert list(split.index) == list(flow_info.index)
    # _pcaps lists newest first: row 0 is the newest pcap (test), row 2 the oldest (train)
    assert split.tolist() == ["test", "test", "val", "train", "train"]


def test_too_few_pcaps_raises_and_names_the_app():
    with pytest.raises(ValueError, match="'Sparse' has 2 pcap"):
        assign_pcap_splits(pd.concat([_pcaps("Fine", 5), _pcaps("Sparse", 2)]))


def test_fractions_that_leave_no_training_pcaps_raise():
    with pytest.raises(ValueError, match="no training pcaps"):
        assign_pcap_splits(_pcaps("App", 4), val_frac=0.4, test_frac=0.4)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="capture_timestamp"):
        assign_pcap_splits(_pcaps("App", 3).drop(columns="capture_timestamp"))
