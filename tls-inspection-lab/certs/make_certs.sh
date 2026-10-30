#!/bin/bash
# Self-signed certificates for the two testbeds (not committed)
cd "$(dirname "$0")"
openssl req -x509 -newkey rsa:2048 -nodes -days 60 -subj "/CN=shop.lab" -keyout server.key -out server.crt
openssl req -x509 -newkey rsa:2048 -nodes -days 60 -subj "/CN=vic2.clouds.web.id" -keyout hikari.key -out server.crt.hikari
