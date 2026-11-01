#!/bin/sh
# Offline replay demo: run Suricata and Zeek on a small capture, then show each flow with both verdicts.
# Usage: demo/replay.sh [xss|dos|hikari|<pcap>]   xss = web attack slice (default), dos = DoS slice, hikari = HIKARI XML-RPC brute force
set -e
cd "$(dirname "$0")/.."
case ${1:-xss} in
    xss) pcap=demo/assets/cic_web_slice.pcap ;;
    dos) pcap=demo/assets/cic_dos_slice.pcap ;;
    hikari) pcap=demo/assets/hikari_xml_slice.pcap; model=hikari ;;
    *) pcap=$1 ;;
esac
out=demo/run
[ -d integration/zeek-flowmeter ] && [ -f integration/et-open-no-engine.rules ] || {
    echo "Run integration/setup.sh once first (needs internet)."; exit 1; }
echo "Replaying $pcap through Suricata (ET Open) and Zeek + zeek-flowmeter ..."
RULES_FILE=integration/et-open-no-engine.rules SURICATA_CONFIG=integration/suricata-alerts-only.yaml \
    integration/run_pcap.sh "$pcap" "$out" >/dev/null 2>&1
python demo/show_flows.py "$out" ${model:-cic}
