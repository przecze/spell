#!/bin/sh
set -e

apk add --no-cache wireguard-tools wireguard-go

if [ ! -s /etc/wireguard/gateway.conf ]; then
  echo "Missing or empty gateway.conf - run 'make wg-create' first"
  exit 1
fi

export WG_QUICK_USERSPACE_IMPLEMENTATION=wireguard-go
export FLY_DNS=$(grep "^DNS" /etc/wireguard/gateway.conf | cut -d= -f2 | tr -d " ")
echo "Fly DNS: $FLY_DNS"

envsubst "\${FLY_DNS}" < /etc/nginx/templates/default.conf > /etc/nginx/conf.d/default.conf

echo "Bringing up WireGuard tunnel (userspace)..."
wg-quick up gateway
echo "WireGuard up, nginx starting..."

exec nginx -g "daemon off;"
