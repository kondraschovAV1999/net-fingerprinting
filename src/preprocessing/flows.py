"""Group parsed packets into bidirectional flows and encode them as fixed-length packet sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

MAX_FRAME_LEN = 1514  # full Ethernet frame, used to scale packet sizes into [-1, 1]

# Per-pcap metadata carried into the flow table when present in the packet table
PCAP_METADATA = ("dataset_name", "session", "capture_timestamp")


def device_ips(packets: pd.DataFrame) -> pd.Series:
    """Capture device IP for each pcap: the address seen in the most packets, as source or destination."""
    endpoints = pd.concat([
        packets[["file_source", "src_ip"]].rename(columns={"src_ip": "ip"}),
        packets[["file_source", "dst_ip"]].rename(columns={"dst_ip": "ip"}),
    ])
    return endpoints.groupby("file_source")["ip"].agg(lambda ips: ips.mode().iloc[0])


def build_flows(packets: pd.DataFrame, min_pkts: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign packets to bidirectional flows and keep flows with at least ``min_pkts`` packets.

    Args:
        packets: One row per packet with columns ``file_source``, ``application``, ``time`` (s),
            ``len`` (bytes), ``protocol``, ``src_ip``, ``dst_ip``, ``src_port`` and ``dst_port``.
            Any of ``PCAP_METADATA`` that are present are copied into the flow table.
        min_pkts: Flows with fewer packets are dropped (e.g. bare TCP handshakes).

    Returns:
        ``(pkt_df, flow_info)``:
            ``pkt_df`` holds the TCP/UDP packets to or from each capture's device that belong to
            kept flows, sorted by ``(flow_id, time)``, with added columns ``direction`` (+1 sent by
            the device, -1 received), ``flow_id``, ``iat_ms`` (time since the flow's previous
            packet) and ``pos`` (packet index within the flow).
            ``flow_info`` has one row per kept flow, indexed by ``flow_id`` in sorted order, with
            ``application``, ``file_source``, ``n_packets``, ``start_time`` and the pcap metadata.
    """
    pkt_df = packets[packets["protocol"].isin(["TCP", "UDP"])].dropna(
        subset=["src_ip", "dst_ip", "src_port", "dst_port"]
    ).copy()
    pkt_df["src_port"] = pkt_df["src_port"].astype(int)
    pkt_df["dst_port"] = pkt_df["dst_port"].astype(int)

    # Keep only traffic to/from each capture's device (the device IP differs between captures)
    device = pkt_df["file_source"].map(device_ips(pkt_df))
    pkt_df = pkt_df[(pkt_df["src_ip"] == device) | (pkt_df["dst_ip"] == device)].copy()
    outbound = pkt_df["src_ip"] == device.loc[pkt_df.index]
    pkt_df["direction"] = np.where(outbound, 1, -1)

    # Both directions of a connection share one (device endpoint, remote endpoint) key
    src_ep = pkt_df["src_ip"] + ":" + pkt_df["src_port"].astype(str)
    dst_ep = pkt_df["dst_ip"] + ":" + pkt_df["dst_port"].astype(str)
    pkt_df["flow_id"] = (pkt_df["file_source"] + "|" + pkt_df["protocol"] + "|"
                         + src_ep.where(outbound, dst_ep) + "|" + dst_ep.where(outbound, src_ep))

    pkt_df = pkt_df.sort_values(["flow_id", "time"], kind="stable")
    flow_sizes = pkt_df.groupby("flow_id").size()
    pkt_df = pkt_df[pkt_df["flow_id"].map(flow_sizes) >= min_pkts].copy()
    pkt_df["iat_ms"] = pkt_df.groupby("flow_id")["time"].diff().fillna(0.0) * 1000
    pkt_df["pos"] = pkt_df.groupby("flow_id").cumcount()

    metadata = ["application", "file_source"] + [c for c in PCAP_METADATA if c in pkt_df.columns]
    by_flow = pkt_df.groupby("flow_id")
    flow_info = by_flow[metadata].first()
    flow_info["n_packets"] = by_flow.size()
    flow_info["start_time"] = by_flow["time"].min()
    return pkt_df, flow_info


def flow_sequences(pkt_df: pd.DataFrame, flow_ids: Sequence[str], seq_len: int = 32) -> np.ndarray:
    """Encode each flow's first ``seq_len`` packets as a ``(n_flows, 2, seq_len)`` float32 array.

    Channel 0 is the packet size signed by direction and scaled by ``MAX_FRAME_LEN`` (clipped,
    in case of offloaded frames larger than the MTU); channel 1 is ``log1p`` of the inter-arrival
    time in ms. Shorter flows are zero-padded; rows follow the order of ``flow_ids``.
    """
    flow_ids = pd.Index(flow_ids)
    seq = pkt_df[pkt_df["flow_id"].isin(flow_ids) & (pkt_df["pos"] < seq_len)]
    rows = flow_ids.get_indexer(seq["flow_id"])
    cols = seq["pos"].to_numpy()

    X = np.zeros((len(flow_ids), 2, seq_len), dtype=np.float32)
    X[rows, 0, cols] = seq["direction"].to_numpy() * np.clip(seq["len"].to_numpy() / MAX_FRAME_LEN, 0, 1)
    X[rows, 1, cols] = np.log1p(seq["iat_ms"].to_numpy())
    return X
