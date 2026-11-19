"""Parse PCAP files into a pandas DataFrame."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader

from utils.schema.parsing import FilenameMetadata, ParsingConfig

logger = logging.getLogger(__name__)


def _packet_to_row(
    idx: int,
    pkt: Any,
    metadata: FilenameMetadata,
    dataset_name: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_name": dataset_name,
        "capture_timestamp": metadata.capture_timestamp,
        "application": metadata.application,
        "download_speed": metadata.download_speed,
        "upload_speed": metadata.upload_speed,
        "file_source": metadata.file_source,
        "packet_num": idx + 1,
        "time": float(pkt.time),
        "len": len(pkt),
        "protocol": "Other",
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
    }
    if pkt.haslayer(IP):
        row["src_ip"] = pkt[IP].src
        row["dst_ip"] = pkt[IP].dst
        if pkt.haslayer(TCP):
            row["protocol"] = "TCP"
            row["src_port"] = pkt[TCP].sport
            row["dst_port"] = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            row["protocol"] = "UDP"
            row["src_port"] = pkt[UDP].sport
            row["dst_port"] = pkt[UDP].dport
        else:
            row["protocol"] = "IP"
    return row


def parse_pcap_to_dataframe(
    pcap_path: Path,
    cfg: ParsingConfig,
    metadata: FilenameMetadata,
    dataset_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        with PcapReader(str(pcap_path)) as reader:
            for idx, pkt in enumerate(reader):
                if idx >= cfg.packet_limit:
                    break
                rows.append(_packet_to_row(idx, pkt, metadata, dataset_name))
    except Exception as exc:  # noqa: BLE001 — skip corrupt/unreadable files
        logger.warning("Failed to read %s: %s", pcap_path, exc)
        return pd.DataFrame()
    return pd.DataFrame(rows)
