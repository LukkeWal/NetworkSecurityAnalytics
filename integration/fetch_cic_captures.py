"""
fetch_cic_captures.py downloads the victim-server and a few client captures of CSE-CIC-IDS2018 from its public bucket.
Only single files are pulled out of the large daily archives. Needs remotezip (pip install remotezip).

    python integration/fetch_cic_captures.py pcaps_cic
"""

import shutil
import sys
from pathlib import Path

from remotezip import RemoteZip

BUCKET = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original%20Network%20Traffic%20and%20Log%20data/"
CAPTURES = {
    "Wednesday-14-02-2018": {"UCAP172.31.69.25": "victim_172.31.69.25.pcap"},
    "Thursday-15-02-2018": {"UCAP172.31.69.25": "victim_172.31.69.25.pcap"},
    "Friday-16-02-2018": {"UCAP172.31.69.25-part1.pcap": "victim_172.31.69.25-part1.pcap"},
    "Thursday-22-02-2018": {"UCAP172.31.69.28": "victim_172.31.69.28.pcap", "UCAP172.31.69.25": "victim_172.31.69.25.pcap"},
    "Friday-23-02-2018": {
        "UCAP172.31.69.28": "victim_172.31.69.28.pcap",
        "UCAP172.31.69.27": "victim_172.31.69.27.pcap",
        **{f"capPC1-172.31.{host}": f"client_172.31.{host}.pcap"
           for host in ["65.117", "66.103", "64.23", "64.21", "64.97", "64.52"]},
    },
}


def main(output: Path) -> None:
    for day, members in CAPTURES.items():
        archive = RemoteZip(f"{BUCKET}{day}/pcap.zip")
        (output / day).mkdir(parents=True, exist_ok=True)
        for member, name in members.items():
            archive.extract(f"pcap/{member}", output / day)
            shutil.move(output / day / "pcap" / member, output / day / name)
            print(f"{day}/{name}")
        shutil.rmtree(output / day / "pcap", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pcaps_cic"))
