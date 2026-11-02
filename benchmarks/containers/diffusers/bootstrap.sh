#!/bin/bash
set -eu
: "${PUBLIC_KEY:?Supply an SSH public key; never a private key}"
ttl="${BENCH_TTL_SECONDS:-3300}"
case "$ttl" in ''|*[!0-9]*) echo "Invalid lease seconds" >&2; exit 1 ;; esac
if [ "$ttl" -lt 1 ] || [ "$ttl" -gt 86400 ]; then
    echo "Lease must be within 1..86400 seconds" >&2
    exit 1
fi
mkdir -p /root/.ssh /run/sshd /workspace/bench
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
printf '\nPasswordAuthentication no\nPermitRootLogin prohibit-password\n' >> /etc/ssh/sshd_config
(sleep "$ttl"; kill -TERM 1) &
exec /usr/sbin/sshd -D -e
