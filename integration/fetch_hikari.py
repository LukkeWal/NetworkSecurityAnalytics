"""
Downloads HIKARI-2021 from Zenodo: the CSV into data/ and the six pcaps into pcaps/ (about 11 GB).

    python integration/fetch_hikari.py
    python integration/fetch_hikari.py --csv-only
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/6463389/files"
CSV_ARCHIVE = "ALLFLOWMETER_HIKARI2021.csv.zip"
PCAPS = [
    "BRUTEFORCE_HTTPS/pcap/Friday_2021-04-16_2304.pcap",
    "BRUTEFORCE_HTTPS/pcap/Sunday_2021-04-11_2154.pcap",
    "BRUTEFORCE_XML/pcap/Monday_2021-04-12_0611.pcap",
    "BRUTEFORCE_XML/pcap/Saturday_2021-04-17_0357.pcap",
    "SCANCMS/pcap/Sunday_2021-05-02_1206.pcap",
    "SCANCMS/pcap/Sunday_2021-05-02_1659.pcap",
]


def download(name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{RECORD}/{name}/content") as response:
        size = int(response.headers["Content-Length"])
        if destination.exists() and destination.stat().st_size == size:
            print(f"have {destination}")
            return
        with open(destination, "wb") as output:
            shutil.copyfileobj(response, output, 1 << 22)
    print(f"downloaded {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HIKARI-2021 from Zenodo.")
    parser.add_argument("--csv-only", action="store_true", help="Skip the pcaps.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory for the CSV (default: data).")
    parser.add_argument("--pcaps", type=Path, default=Path("pcaps"), help="Directory for the pcaps (default: pcaps).")
    args = parser.parse_args()

    archive = args.data / CSV_ARCHIVE
    download(CSV_ARCHIVE, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extract("ALLFLOWMETER_HIKARI2021.csv", args.data)
    archive.unlink()
    if not args.csv_only:
        for name in PCAPS:
            download(name, args.pcaps / name)


if __name__ == "__main__":
    main()
