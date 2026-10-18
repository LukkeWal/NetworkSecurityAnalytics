#!/bin/sh
# Usage: lab.sh <build|up|run|stop|status|down> <run id> [hikari]
# Every run id is a separate copy of the lab (own containers, network and runs/<id>/ folder).
set -e
cd "$(dirname "$0")"
command=${1:?usage: lab.sh <build|up|run|stop|status|down> <run id>}
run=${2:?run id missing}
case $run in *[!a-z0-9-]*) echo "run id: lowercase letters, digits and dashes only"; exit 1;; esac
export LAB_RUN=$run
compose() { docker compose -p "tlslab-$run" "$@"; }
runs=runs/$run

ip_of() { docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$(compose ps -q "$1")"; }

case $command in
build)
    docker image inspect tlslab-runner:1 >/dev/null 2>&1 || docker build -t tlslab-runner:1 runner
    ;;
up)
    mkdir -p "$runs/captures"
    "$0" build "$run"
    compose up -d db wordpress tls sniffer browser_attacker browser_benign runner scanner
    until compose --profile tools run --rm -T wpcli wp core install --url=https://tls --title="Lab Shop" \
        --admin_user=admin --admin_password='Xq7-lab-Tr1cky-9fW' --admin_email=admin@lab.test --skip-email 2>/dev/null \
        || compose --profile tools run --rm -T wpcli wp core is-installed 2>/dev/null; do sleep 4; done
    compose --profile tools run --rm -T wpcli wp user get alice >/dev/null 2>&1 \
        || compose --profile tools run --rm -T wpcli wp user create alice alice@lab.test --role=author --user_pass='Alice-lab-Pw-4421'
    compose --profile tools run --rm -T wpcli wp post list --format=count | grep -q '^[1-9]' \
        || compose --profile tools run --rm -T wpcli wp post create --post_title="Welcome to the shop" --post_content="Hello from the lab." --post_status=publish
    printf '{"browser_attacker": "%s", "runner": "%s", "scanner": "%s", "browser_benign": "%s"}\n' \
        "$(ip_of browser_attacker)" "$(ip_of runner)" "$(ip_of scanner)" "$(ip_of browser_benign)" > "$runs/clients.json"
    cat "$runs/clients.json"
    ;;
run)
    log=$runs/scenarios.log
    : > "$log"
    # "hikari" asks for about as many flows as HIKARI-2021 has on port 443 (34k attack, 82k benign)
    if [ "${3:-}" = hikari ]; then
        BENIGN_SESSIONS=4 BENIGN_ROUNDS=190 BRUTE_SESSIONS=2 BRUTE_ATTEMPTS=490 XMLRPC_ATTEMPTS=5145 WPSCAN_RUNS=13 SLEEP_SCALE=0.3
    fi
    timed() { name=$1; shift; start=$(date +%s); "$@" || echo "$name failed" >&2; echo "$name,$start,$(date +%s)" >> "$log"; }
    wpscan_runs() {
        n=0
        while [ "$n" -lt "${WPSCAN_RUNS:-1}" ]; do
            compose exec -T scanner wpscan --url https://tls/ --disable-tls-checks --no-update -e p,t,u >/dev/null 2>&1 || true
            n=$((n + 1))
        done
    }
    i=0
    while [ "$i" -lt "${BENIGN_SESSIONS:-1}" ]; do
        timed "benign$i" compose exec -T -e SLEEP_SCALE="${SLEEP_SCALE:-1}" runner python /attacks/benign_browsing.py http://browser_benign:4444 "${BENIGN_ROUNDS:-12}" &
        i=$((i + 1))
    done
    i=0
    while [ "$i" -lt "${BRUTE_SESSIONS:-1}" ]; do
        timed "browser_bruteforce$i" compose exec -T runner python /attacks/browser_login_bruteforce.py http://browser_attacker:4444 /attacks/10k-most-common.txt "${BRUTE_ATTEMPTS:-150}" $((i * ${BRUTE_ATTEMPTS:-150})) &
        i=$((i + 1))
    done
    timed xmlrpc_bruteforce compose exec -T runner python /attacks/xmlrpc_bruteforce.py /attacks/10k-most-common.txt "${XMLRPC_ATTEMPTS:-600}" &
    timed wpscan wpscan_runs &
    wait
    cat "$log"
    ;;
stop)
    compose stop sniffer
    ls -la "$runs/captures"
    ;;
status)
    compose ps
    ;;
down)
    compose down -v
    ;;
*)
    echo "unknown command $command"; exit 1
    ;;
esac
