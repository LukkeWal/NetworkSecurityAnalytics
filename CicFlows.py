"""
Reads the flows Zeek rebuilds from CSE-CIC-IDS2018 captures and labels them. The captures have no labels, so a flow counts
as an attack when its source is an attacker address documented for that day.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from ZeekFlows import CONN_COLUMNS

ATTACKERS = {
    "18.221.219.4": "FTP-BruteForce",
    "13.58.98.64": "SSH-BruteForce",
    "18.219.211.138": "DoS-GoldenEye",
    "18.217.165.70": "DoS-Slowloris",
    "13.59.126.31": "DoS-SlowHTTPTest",
    "18.219.193.20": "DoS-Hulk",
    "18.218.115.60": "Web attack",
}
WEB_ATTACKS = ["Web-BruteForce", "XSS", "SQLi"]
BURST_GAP_SECONDS = 30 * 60
CHUNK_ROWS = 200_000
# zeek-flowmeter wraps around to about 2^64 on a few flows with out-of-order packets; pandas cannot parse such values
OVERFLOWED_COUNT = re.compile(r"(?<=[:,\[])-?\d{19,}(\.\d+)?(?=[,}\]])")


def read_labelled(zeek_dir: Path | str, max_per_attack: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Rebuilt flows with Label and traffic_category. All benign flows are kept, each attack type is sampled to max_per_attack."""
    zeek_dir = Path(zeek_dir)
    conn = _read_in_chunks(zeek_dir / "conn.log", ["uid", "ts", "proto", *CONN_COLUMNS]).rename(columns=CONN_COLUMNS)
    conn["traffic_category"] = conn["originh"].map(ATTACKERS).fillna("Benign")
    _name_web_attacks(conn)

    keep = conn[conn["traffic_category"] == "Benign"]
    for _, flows in conn[conn["traffic_category"] != "Benign"].groupby("traffic_category"):
        keep = pd.concat([keep, flows.sample(min(len(flows), max_per_attack), random_state=random_state)])
    keep = keep.set_index("uid")

    chunks = []
    for chunk in _json_chunks(zeek_dir / "flowmeter.log"):
        chunks.append(chunk[chunk["uid"].isin(keep.index)])
    flows = pd.concat(chunks).join(keep, on="uid")
    flows["Label"] = (flows["traffic_category"] != "Benign").astype(int)
    return flows.reset_index(drop=True)


def _read_in_chunks(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.concat(chunk[columns] for chunk in _json_chunks(path))


def _json_chunks(path: Path):
    with open(path) as log:
        while True:
            lines = [line for _, line in zip(range(CHUNK_ROWS), log)]
            if not lines:
                return
            text = OVERFLOWED_COUNT.sub("null", "".join(lines))
            yield pd.read_json(io.StringIO(text), lines=True, convert_dates=False)


def _name_web_attacks(conn: pd.DataFrame) -> None:
    """The web attacker runs brute force, XSS and SQL injection in turn, separated by long gaps."""
    web = conn[conn["traffic_category"] == "Web attack"].sort_values("ts")
    if web.empty:
        return
    burst = (web["ts"].diff() > BURST_GAP_SECONDS).cumsum()
    names = burst.map(lambda number: WEB_ATTACKS[number] if number < len(WEB_ATTACKS) else "Web attack")
    conn.loc[web.index, "traffic_category"] = names
