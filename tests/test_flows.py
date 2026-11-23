import numpy as np
import pandas as pd
import pytest

from preprocessing.flows import MAX_FRAME_LEN, build_flows, device_ips, flow_sequences

DEVICE = "192.168.137.42"
SERVER = "3.160.212.116"
PCAP = "1723117813_Duolingo_22.66-12.56.pcap"


def _pkt(t, length, src, dst, sport, dport, protocol="TCP", file_source=PCAP, application="Duolingo"):
    return {"file_source": file_source, "application": application, "time": t, "len": length,
            "protocol": protocol, "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport}


@pytest.fixture
def packets():
    rows = [
        # Flow A: 5 packets in both directions between the device and the server
        _pkt(0.000, 74, DEVICE, SERVER, 50000, 443),
        _pkt(0.010, 74, SERVER, DEVICE, 443, 50000),
        _pkt(0.012, 66, DEVICE, SERVER, 50000, 443),
        _pkt(0.050, 1514, SERVER, DEVICE, 443, 50000),
        _pkt(0.051, 3000, SERVER, DEVICE, 443, 50000),  # offloaded frame larger than the MTU
        # Flow B: only 2 packets -> dropped with min_pkts=4
        _pkt(1.000, 74, DEVICE, SERVER, 50001, 443),
        _pkt(1.010, 74, SERVER, DEVICE, 443, 50001),
        # Traffic that doesn't involve the device -> dropped
        _pkt(2.000, 100, "192.168.137.1", "8.8.8.8", 53000, 53, protocol="UDP"),
        _pkt(2.001, 100, "8.8.8.8", "192.168.137.1", 53, 53000, protocol="UDP"),
        # Non-TCP/UDP packet without IPs -> dropped
        _pkt(3.000, 60, None, None, None, None, protocol="Other"),
    ]
    return pd.DataFrame(rows)


def test_device_ip_is_the_most_frequent_address(packets):
    assert device_ips(packets.dropna(subset=["src_ip"])).to_dict() == {PCAP: DEVICE}


def test_both_directions_share_one_flow_and_short_flows_are_dropped(packets):
    pkt_df, flow_info = build_flows(packets, min_pkts=4)
    assert list(flow_info.index) == [f"{PCAP}|TCP|{DEVICE}:50000|{SERVER}:443"]
    assert len(pkt_df) == 5
    assert pkt_df["direction"].tolist() == [1, -1, 1, -1, -1]


def test_iat_and_position_within_flow(packets):
    pkt_df, _ = build_flows(packets, min_pkts=4)
    assert pkt_df["pos"].tolist() == [0, 1, 2, 3, 4]
    assert pkt_df["iat_ms"].to_numpy() == pytest.approx([0.0, 10.0, 2.0, 38.0, 1.0])


def test_flow_info_summarises_flows_and_carries_pcap_metadata(packets):
    packets["dataset_name"] = "Iraq"
    packets["session"] = "Iraq-36"
    packets["capture_timestamp"] = 1723117813
    _, flow_info = build_flows(packets, min_pkts=4)
    row = flow_info.iloc[0]
    assert row["application"] == "Duolingo"
    assert row["file_source"] == PCAP
    assert (row["dataset_name"], row["session"], row["capture_timestamp"]) == ("Iraq", "Iraq-36", 1723117813)
    assert row["n_packets"] == 5
    assert row["start_time"] == pytest.approx(0.0)


def test_lower_min_pkts_keeps_short_flows(packets):
    _, flow_info = build_flows(packets, min_pkts=2)
    assert len(flow_info) == 2


def test_float_ports_from_csv_give_the_same_flow_id(packets):
    as_float = packets.astype({"src_port": float, "dst_port": float})
    assert list(build_flows(as_float)[1].index) == list(build_flows(packets)[1].index)


def test_flow_sequences_encode_size_direction_and_timing(packets):
    pkt_df, flow_info = build_flows(packets, min_pkts=4)
    X = flow_sequences(pkt_df, flow_info.index, seq_len=8)
    assert X.shape == (1, 2, 8) and X.dtype == np.float32
    sizes = [74, -74, 66, -1514, -MAX_FRAME_LEN]  # the 3000-byte frame is clipped to the max frame size
    assert X[0, 0, :5] == pytest.approx(np.array(sizes) / MAX_FRAME_LEN)
    assert X[0, 1, :5] == pytest.approx(np.log1p([0.0, 10.0, 2.0, 38.0, 1.0]))
    assert not X[0, :, 5:].any()  # zero padding after the last packet


def test_flow_sequences_truncate_and_follow_requested_order(packets):
    pkt_df, flow_info = build_flows(packets, min_pkts=2)
    order = list(flow_info.index[::-1])
    X = flow_sequences(pkt_df, order, seq_len=3)
    assert X.shape == (2, 2, 3)
    expected_first = flow_sequences(pkt_df, [order[0]], seq_len=3)[0]
    assert np.array_equal(X[0], expected_first)
    assert X[1, 0].tolist() == pytest.approx([74 / MAX_FRAME_LEN, -74 / MAX_FRAME_LEN, 66 / MAX_FRAME_LEN])
