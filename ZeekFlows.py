"""
Reads the flows Zeek and zeek-flowmeter rebuild from a pcap. flowmeter gives the same feature columns as the HIKARI-2021 CSV,
so the flows go straight into the model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENDPOINT_COLUMNS = ["originh", "originp", "responh", "responp"]
DURATION_TOLERANCE = 1e-6
CONN_COLUMNS = {
    "id.orig_h": "originh",
    "id.orig_p": "originp",
    "id.resp_h": "responh",
    "id.resp_p": "responp",
}


def run_folders(paths: list[Path]) -> list[Path]:
    """Each path is a run folder (with a zeek folder inside) or a folder holding run folders."""
    folders = []
    for path in map(Path, paths):
        folders += [path] if (path / "zeek").is_dir() else sorted(p for p in path.iterdir() if (p / "zeek").is_dir())
    return folders


def read_flows(zeek_dir: Path | str) -> pd.DataFrame:
    """Join flowmeter.log and conn.log on uid, giving features plus endpoints, start time and protocol."""
    zeek_dir = Path(zeek_dir)
    flowmeter = pd.read_json(zeek_dir / "flowmeter.log", lines=True, convert_dates=False)
    conn = pd.read_json(zeek_dir / "conn.log", lines=True, convert_dates=False)
    conn = conn[["uid", "ts", "proto", *CONN_COLUMNS]].rename(columns=CONN_COLUMNS)
    return flowmeter.merge(conn, on="uid")


def attach_labels(flows: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """
    Copy Label, traffic_category and row_id from the HIKARI CSV onto the rebuilt flows, matching on endpoints, packet counts and
    duration (the CSV has no timestamps). A flow listed as both attack and Benign gets the attack.
    """
    keys = ENDPOINT_COLUMNS + ["fwd_pkts_tot", "bwd_pkts_tot"]
    truth = labelled[[*keys, "flow_duration", "Label", "traffic_category", "row_id"]]
    truth = truth.sort_values("Label", ascending=False).drop_duplicates([*keys, "flow_duration"])
    return pd.merge_asof(
        flows.sort_values("flow_duration"),
        truth.sort_values("flow_duration"),
        on="flow_duration",
        by=keys,
        direction="nearest",
        tolerance=DURATION_TOLERANCE,
    )
