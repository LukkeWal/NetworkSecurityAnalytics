#!/bin/sh
# Run the authors' autobrowse.py (random top-1M sites, Firefox then Chrome) repeatedly for a number of minutes.
# Only SAMPLE is changed, from 10000 sites to a few, so that each pass finishes in minutes.
minutes=${1:?minutes}
sites=${2:-12}
end=$(( $(date +%s) + minutes * 60 ))
cd /lab/profiles/benign
sed "s/^SAMPLE=10000/SAMPLE=$sites/" autobrowse.py > "/tmp/autobrowse_$$.py"
while [ "$(date +%s)" -lt "$end" ]; do
    timeout $(( end - $(date +%s) )) python "/tmp/autobrowse_$$.py" > /dev/null 2>&1 || true
done
