#!/bin/sh
# Replays a saved lab capture (no Docker needed, about 1 minute): Zeek, Suricata and the models analyse the recorded traffic.
# The capture holds the encrypted side (port 443) and the plain hop behind nginx, so both views are available.
# Usage: demo/lab_replay.sh
set -e
cd "$(dirname "$0")/../tls-inspection-lab"
run=replay
rm -rf "runs/$run"
mkdir -p "runs/$run/captures"
cp ../demo/assets/lab_slice.pcap "runs/$run/captures/lab.pcap"
cp ../demo/assets/lab_slice_clients.json "runs/$run/clients.json"
echo "Analysing the capture with Zeek, Suricata and the models (about a minute)..."
python analyze.py "$run" | grep -v '^\[timing\]'
