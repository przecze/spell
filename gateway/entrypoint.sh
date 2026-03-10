#!/bin/sh
set -e

if [ ! -s /etc/wireguard/gateway.conf ]; then
  echo "Missing or empty gateway.conf - run 'make wg-create' first"
  exit 1
fi

export WG_QUICK_USERSPACE_IMPLEMENTATION=wireguard-go
FLY_DNS=$(grep "^DNS" /etc/wireguard/gateway.conf | cut -d= -f2 | tr -d " ")
echo "Fly DNS: $FLY_DNS"

# Template variables (see nginx.conf.template for docs)
export RESOLVER="[$FLY_DNS] valid=30s"
export BACKEND_HOST="api-spell-jan-czechowski-com.flycast"
export DATASET_HOST="dataset-spell-jan-czechowski-com.flycast"
export VITE_UPSTREAM="localhost:1"  # never reached — baked frontend always serves first
export ACTIVITY_LOG="/activity/most_recent_call.txt"
export FRONTEND_CSP="default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self' data:; img-src 'self' data: https:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none';"

envsubst '${RESOLVER} ${BACKEND_HOST} ${DATASET_HOST} ${VITE_UPSTREAM} ${ACTIVITY_LOG} ${FRONTEND_CSP}' \
  < /etc/nginx/templates/nginx.conf.template \
  > /etc/nginx/conf.d/default.conf

echo "Bringing up WireGuard tunnel (userspace)..."
wg-quick up gateway
echo "WireGuard up, nginx starting..."

exec nginx -g "daemon off;"
