#!/bin/sh
# One-off setup: fetch zeek-flowmeter and the ET Open ruleset.
set -e
cd "$(dirname "$0")"

[ -d zeek-flowmeter ] || git clone --depth 1 https://github.com/zeek-flowmeter/zeek-flowmeter.git zeek-flowmeter
suricata-update

# Same rules without Suricata's own packet-level events (bad checksums etc.), which only bloat the logs.
# Use it with RULES_FILE=integration/et-open-no-engine.rules integration/run_pcap.sh ...
grep -v 'msg:"SURICATA ' "${HOMEBREW_PREFIX:-/opt/homebrew}/var/lib/suricata/rules/suricata.rules" > et-open-no-engine.rules
