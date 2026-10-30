#!/bin/sh
cp /certs/server.crt /usr/local/share/ca-certificates/lab.crt 2>/dev/null && update-ca-certificates >/dev/null 2>&1
mitmdump --listen-port 8080 --set ssl_insecure=true -q > /tmp/mitmdump.log 2>&1 &
exec sleep infinity
