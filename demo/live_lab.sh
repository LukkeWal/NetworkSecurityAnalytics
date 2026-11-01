#!/bin/sh
# Runs the TLS lab for real (needs Docker Desktop, about 3 GB of memory, about 6 minutes), then analyses the capture.
# Small traffic volumes, so the percentages are noisier than a full run.
# Usage: demo/live_lab.sh [run-id]
set -e
cd "$(dirname "$0")/../tls-inspection-lab"
run=${1:-demo}
rm -rf "runs/$run"
./lab.sh up "$run"
BENIGN_ROUNDS=3 BRUTE_ATTEMPTS=30 XMLRPC_ATTEMPTS=100 ./lab.sh run "$run"
./lab.sh stop "$run"
./lab.sh down "$run"
echo "Lab stopped. Analysing the capture with Zeek, Suricata and the models (about a minute)..."
python analyze.py "$run" | grep -v '^\[timing\]'
