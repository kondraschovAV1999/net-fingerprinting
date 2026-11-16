import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_extraction import (
    FEATURE_CONFIGS,
    FeatureGroups,
    FlowFeatureExtractor,
    detect_client_ip,
    make_extractor,
    prepare_packet_dataframe,
)


def _flow_packets(flow_id="f0", n_out=3, n_in=3):
    rows = []
    t = 0.0
    for i in range(n_out):
        rows.append(
            {"flow_id": flow_id, "time": t, "len": 100 + 10 * i, "direction": 1}
        )
        t += 0.5
    for i in range(n_in):
        rows.append({"flow_id": flow_id, "time": t, "len": 60 + 5 * i, "direction": -1})
        t += 0.4
    return pd.DataFrame(rows)


def _flow_df():
    return _flow_packets()


def test_foap_feature_count():
    X = FlowFeatureExtractor().transform(_flow_df())
    assert X.shape == (1, 123)


def test_basic_feature_count():
    X = FlowFeatureExtractor(groups=FEATURE_CONFIGS["basic"]).transform(_flow_df())
    assert X.shape == (1, 8)


def test_statistical_feature_count():
    X = FlowFeatureExtractor(groups=FEATURE_CONFIGS["statistical"]).transform(
        _flow_df()
    )
    assert X.shape == (1, 103)


def test_transform_groups_flows():
    df = pd.concat([_flow_packets("a"), _flow_packets("b")], ignore_index=True)
    X = FlowFeatureExtractor().transform(df)
    assert X.shape == (2, 123)
    assert list(X.index) == ["a", "b"]


def test_flow_meta_populated():
    df = _flow_df().assign(
        file_source="1712045358_Telegram_3.69-3.11_.pcap",
        application="Telegram",
        capture_timestamp=1712045358,
    )
    ext = FlowFeatureExtractor()
    ext.transform(df)
    assert ext.flow_meta is not None
    assert list(ext.flow_meta.columns) == [
        "file_source",
        "application",
        "capture_timestamp",
    ]
    assert ext.flow_meta.loc["f0", "application"] == "Telegram"


def test_flow_meta_none_when_absent():
    ext = FlowFeatureExtractor()
    ext.transform(_flow_df())
    assert ext.flow_meta is None


def test_unidirectional_flow_has_no_nan():
    df = _flow_packets(n_in=0)
    X = FlowFeatureExtractor().transform(df)
    assert not X.isna().to_numpy().any()


def test_make_extractor_presets():
    assert make_extractor("foap").groups == FEATURE_CONFIGS["foap"]
    assert make_extractor("basic").groups == FEATURE_CONFIGS["basic"]
    assert make_extractor("statistical").groups == FEATURE_CONFIGS["statistical"]
    with pytest.raises(ValueError):
        make_extractor("nope")


def test_prepare_packet_dataframe_bidirectional_flow():
    df = pd.DataFrame(
        [
            {
                "src_ip": "192.168.1.5",
                "dst_ip": "8.8.8.8",
                "src_port": 50000,
                "dst_port": 443,
                "protocol": "TCP",
                "time": 0.0,
                "len": 100,
            },
            {
                "src_ip": "8.8.8.8",
                "dst_ip": "192.168.1.5",
                "src_port": 443,
                "dst_port": 50000,
                "protocol": "TCP",
                "time": 0.1,
                "len": 80,
            },
        ]
    )
    out = prepare_packet_dataframe(df)
    assert out["flow_id"].nunique() == 1
    assert list(out["direction"]) == [1, -1]


def test_prepare_packet_dataframe_drops_null_ip_rows():
    df = pd.DataFrame(
        [
            {
                "src_ip": "192.168.1.5",
                "dst_ip": "8.8.8.8",
                "src_port": 50000,
                "dst_port": 443,
                "protocol": "TCP",
            },
            {
                "src_ip": None,
                "dst_ip": None,
                "src_port": None,
                "dst_port": None,
                "protocol": "Other",
            },
        ]
    )
    out = prepare_packet_dataframe(df)
    assert len(out) == 1


def test_detect_client_ip_prefers_private():
    df = pd.DataFrame(
        {
            "src_ip": ["192.168.1.5", "8.8.8.8"],
            "dst_ip": ["8.8.8.8", "192.168.1.5"],
        }
    )
    assert detect_client_ip(df) == "192.168.1.5"
