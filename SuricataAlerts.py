"""
SuricataAlerts reads the alerts Suricata wrote to eve.json and marks which rebuilt Zeek flows triggered one.
A flow counts as flagged by Suricata when an alert matches its endpoints inside its time window.
"""

from __future__ import annotations

import ipaddress
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE_EVENT_PREFIX = "SURICATA "
TIME_SLACK_SECONDS = 1.0


def read_alerts(eve_path: Path | str, ignore_engine_events: bool = True) -> pd.DataFrame:
    """Load alert events from eve.json, without Suricata's own decoder and stream events unless asked."""
    rows = []
    with open(eve_path) as eve_file:
        for line in eve_file:
            if '"event_type":"alert"' not in line:
                continue
            if ignore_engine_events and f'"signature":"{ENGINE_EVENT_PREFIX}' in line:
                continue
            event = json.loads(line)
            signature = event["alert"]["signature"]
            if ignore_engine_events and signature.startswith(ENGINE_EVENT_PREFIX):
                continue
            rows.append((
                event["timestamp"], event["src_ip"], event.get("src_port", 0),
                event["dest_ip"], event.get("dest_port", 0), event["proto"],
                signature, event["alert"]["category"],
            ))
    alerts = pd.DataFrame(
        rows, columns=["timestamp", "src_ip", "src_port", "dest_ip", "dest_port", "proto", "signature", "category"]
    )
    parsed = pd.to_datetime(alerts["timestamp"], utc=True)
    alerts["ts"] = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return alerts


def flag_flows(flows: pd.DataFrame, alerts: pd.DataFrame, slack: float = TIME_SLACK_SECONDS) -> pd.DataFrame:
    """Add suricata_alerts (count), suricata_signature (first one) and suricata_flag to the flows."""
    flows = flows.copy()
    flows["flow_key"] = _flow_key(flows, "originh", "originp", "responh", "responp")
    alerts = alerts.assign(flow_key=_flow_key(alerts, "src_ip", "src_port", "dest_ip", "dest_port"))

    candidates = flows[["flow_key", "ts", "flow_duration"]].reset_index().merge(
        alerts[["flow_key", "ts", "signature"]], on="flow_key", suffixes=("", "_alert")
    )
    inside = candidates[
        (candidates["ts_alert"] >= candidates["ts"] - slack)
        & (candidates["ts_alert"] <= candidates["ts"] + candidates["flow_duration"] + slack)
    ]
    per_flow = inside.groupby("index").agg(
        suricata_alerts=("signature", "size"), suricata_signature=("signature", "first")
    )
    flows = flows.join(per_flow)
    flows["suricata_alerts"] = flows["suricata_alerts"].fillna(0).astype(int)
    flows["suricata_flag"] = flows["suricata_alerts"] > 0
    return flows.drop(columns="flow_key")


def _flow_key(data: pd.DataFrame, src_ip: str, src_port: str, dst_ip: str, dst_port: str) -> pd.Series:
    """Direction-independent key, since an alert can come from either side of the flow."""
    source = data[src_ip].map(_normalise_ip) + ":" + data[src_port].astype(str)
    destination = data[dst_ip].map(_normalise_ip) + ":" + data[dst_port].astype(str)
    first = np.where(source < destination, source, destination)
    second = np.where(source < destination, destination, source)
    return pd.Series(first + "|" + second, index=data.index)


@lru_cache(maxsize=None)
def _normalise_ip(address: str) -> str:
    # Suricata writes IPv6 addresses expanded, Zeek compressed
    return ipaddress.ip_address(address).compressed
