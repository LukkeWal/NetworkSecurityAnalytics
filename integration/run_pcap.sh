#!/bin/sh
# Usage: run_pcap.sh <pcap> [output dir]
# Rebuilds the flows with Zeek + zeek-flowmeter and runs Suricata on the same pcap.
set -e
here=$(cd "$(dirname "$0")" && pwd)
pcap=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
out=${2:-$here/runs/$(basename "$pcap" .pcap)}
prefix=${HOMEBREW_PREFIX:-/opt/homebrew}

# The default HOME_NET is the private ranges, which never match HIKARI's public addresses
home_net=${HOME_NET:-any}
external_net=${EXTERNAL_NET:-any}

# Suricata appends to an existing eve.json, so start from an empty folder
rm -rf "$out/suricata"
mkdir -p "$out/zeek" "$out/suricata"

# -C: the pcaps have bad checksums. SKIP_ZEEK=1 reuses the logs already there.
if [ -z "$SKIP_ZEEK" ]; then
    # some captures end in a half-written packet, which makes zeek stop with an error; its logs are still usable
    (cd "$out/zeek" && zeek -C -r "$pcap" "$here/zeek-flowmeter/scripts" LogAscii::use_json=T) || echo "zeek stopped early"
fi

suricata -c "${SURICATA_CONFIG:-$prefix/etc/suricata/suricata.yaml}" \
    -S "${RULES_FILE:-$prefix/var/lib/suricata/rules/suricata.rules}" \
    --set "vars.address-groups.HOME_NET=$home_net" \
    --set "vars.address-groups.EXTERNAL_NET=$external_net" \
    -k none -r "$pcap" -l "$out/suricata"
