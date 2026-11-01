#!/bin/sh
# Usage: hikari.sh <up|run|stop|status|down> <run id> [hikari]
# The HIKARI-style testbed (docker-compose.hikari.yml): the authors' scripts run against a WordPress site on TLS.
set -e
cd "$(dirname "$0")"
command=${1:?usage: hikari.sh <up|run|stop|status|down> <run id>}
run=${2:?run id missing}
case $run in *[!a-z0-9-]*) echo "run id: lowercase letters, digits and dashes only"; exit 1;; esac
export LAB_RUN=$run
compose() { docker compose -f docker-compose.hikari.yml -p "tlshk-$run" "$@"; }
runs=runs/$run
site=https://vic2.clouds.web.id

ip_of() { docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$(compose ps -q "$1")"; }
wp() { compose --profile tools run --rm -T wpcli wp "$@"; }

case $command in
up)
    mkdir -p "$runs/captures"
    docker image inspect tlslab-hikari:1 >/dev/null 2>&1 || docker build -t tlslab-hikari:1 runner-hikari
    compose up -d db wordpress tls sniffer_server attacker_browser attacker_xmlrpc benign sniffer_benign scanner
    until wp core install --url=$site --title="Lab Shop" --admin_user=admin --admin_password='Xq7-lab-Tr1cky-9fW' \
        --admin_email=admin@lab.test --skip-email 2>/dev/null || wp core is-installed 2>/dev/null; do sleep 4; done
    wp option update users_can_register 1 >/dev/null
    wp option update default_role subscriber >/dev/null
    printf '{"attacker_browser": "%s", "attacker_xmlrpc": "%s", "benign": "%s", "scanner": "%s", "tls": "%s"}\n' \
        "$(ip_of attacker_browser)" "$(ip_of attacker_xmlrpc)" "$(ip_of benign)" "$(ip_of scanner)" "$(ip_of tls)" > "$runs/clients.json"
    cat "$runs/clients.json"
    ;;
run)
    log=$runs/scenarios.log
    : > "$log"
    # "hikari" asks for roughly the size of HIKARI-2021's port-443 traffic
    if [ "${3:-}" = hikari ]; then
        BRUTE_PROCS=3 BRUTE_ATTEMPTS=150 XMLRPC_ATTEMPTS=5145 SIGNUP_PROCS=2 SIGNUPS=200 BROWSE_PROCS=3 BROWSE_MINUTES=90 WPSCAN_RUNS=8
    fi
    timed() { name=$1; shift; start=$(date +%s); "$@" || echo "$name failed" >&2; echo "$name,$start,$(date +%s)" >> "$log"; }
    wpscan_runs() {
        n=0
        while [ "$n" -lt "${WPSCAN_RUNS:-1}" ]; do
            compose exec -T scanner wpscan --url $site/ --disable-tls-checks --no-update -e p,t,u >/dev/null 2>&1 || true
            n=$((n + 1))
        done
    }
    i=0
    while [ "$i" -lt "${BRUTE_PROCS:-1}" ]; do
        timed "browser_bruteforce$i" compose exec -T -w /lab/profiles/malicious attacker_browser python /drivers/bruteforce_browser.py $site /attacks/10k-most-common.txt "${BRUTE_ATTEMPTS:-20}" $((i * ${BRUTE_ATTEMPTS:-20})) &
        i=$((i + 1))
    done
    timed xmlrpc_bruteforce compose exec -T -w /lab/profiles/malicious attacker_xmlrpc python /drivers/bruteforce_xmlrpc.py $site/xmlrpc.php /attacks/10k-most-common.txt "${XMLRPC_ATTEMPTS:-200}" 0 &
    i=0
    while [ "$i" -lt "${SIGNUP_PROCS:-1}" ]; do
        timed "signup$i" compose exec -T -w /lab/profiles/benign benign python /drivers/benign_signup.py "$site/wp-login.php?action=register" "${SIGNUPS:-10}" &
        i=$((i + 1))
    done
    i=0
    while [ "$i" -lt "${BROWSE_PROCS:-1}" ]; do
        timed "browse$i" compose exec -T benign sh /drivers/benign_browse.sh "${BROWSE_MINUTES:-5}" 12 &
        i=$((i + 1))
    done
    timed wpscan wpscan_runs &
    wait
    cat "$log"
    ;;
stop)
    compose stop sniffer_server sniffer_benign
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
