#!/bin/sh
# Demo launcher. Usage: ./demo.sh <check|xss|dos|hikari|lab|lab-live>
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] && PATH="$PWD/.venv/bin:$PATH"
case ${1:-} in
    check)          python smoke_test.py ;;
    xss|dos|hikari) demo/replay.sh "$1" ;;
    lab)            demo/lab_replay.sh ;;
    lab-live)       demo/live_lab.sh ;;
    *)              echo "usage: ./demo.sh <check|xss|dos|hikari|lab|lab-live>"; exit 1 ;;
esac
