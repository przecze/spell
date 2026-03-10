#!/bin/sh
set -e

apk add --no-cache curl jq

FLY_API_TOKEN=$(cat /run/secrets/fly_token)
API="https://api.machines.dev/v1/apps/${FLY_APP_NAME}/machines"
AUTH="Authorization: Bearer ${FLY_API_TOKEN}"
MACHINES_CACHE="/activity/${FLY_APP_NAME}_machines_state.json"
JUST_SUSPENDED="/activity/${FLY_APP_NAME}_just_suspended"

MOST_RECENT_CALL_CHECK_FREQUENCY_S=${MOST_RECENT_CALL_CHECK_FREQUENCY_S:-5}
MOST_RECENT_CALL_MAX_AGE_S=${MOST_RECENT_CALL_MAX_AGE_S:-60}
MACHINES_STATE_MAX_AGE_S=${MACHINES_STATE_MAX_AGE_S:-900}
MACHINE_STARTED_MAX_NOTIFY_S=${MACHINE_STARTED_MAX_NOTIFY_S:-3600}

while true; do
  sleep "${MOST_RECENT_CALL_CHECK_FREQUENCY_S}"
  now=$(date +%s)

  need_fresh=false
  if [ ! -f "${MACHINES_CACHE}" ]; then
    need_fresh=true
  elif [ -f "${JUST_SUSPENDED}" ]; then
    need_fresh=true
    rm -f "${JUST_SUSPENDED}"
  else
    cache_mtime=$(stat -c %Y "${MACHINES_CACHE}" 2>/dev/null || echo 0)
    cache_age=$((now - cache_mtime))
    if [ "${cache_age}" -gt "${MACHINES_STATE_MAX_AGE_S}" ]; then
      need_fresh=true
    fi
  fi

  if [ "${need_fresh}" = true ]; then
    echo "debug: refetching machines state from API"
    json=$(curl -sf -H "${AUTH}" "${API}")
    echo "${json}" | jq . > "${MACHINES_CACHE}"
    echo "debug: machines state saved to ${MACHINES_CACHE}"

    echo "${json}" | jq -r '.[] | select(.state == "started") | [.id, .name, ([.events[]? | select(.type == "start" and .status == "started") | .timestamp] | max // 0)] | @tsv' | while IFS=$(printf '\t') read -r mid mname start_ts; do
      [ -z "${mid}" ] && continue
      [ "${start_ts}" = "0" ] || [ -z "${start_ts}" ] && continue
      loop_now=$(date +%s)
      start_s=$((start_ts / 1000))
      uptime_s=$((loop_now - start_s))
      if [ "${uptime_s}" -gt "${MACHINE_STARTED_MAX_NOTIFY_S}" ]; then
        notified_file="/activity/${FLY_APP_NAME}_notified_${mid}"
        if [ -f "${notified_file}" ]; then
          prev_start_ts=$(cat "${notified_file}" 2>/dev/null)
          [ "${prev_start_ts}" = "${start_ts}" ] && continue
        fi
        started_at=$(TZ=Europe/Warsaw date -d "@${start_s}" '+%Y-%m-%d %H:%M:%S %Z')
        echo "debug: sending notify for machine ${mname} (${mid}), uptime ${uptime_s}s > ${MACHINE_STARTED_MAX_NOTIFY_S}s"
        /notify.sh "${FLY_APP_NAME}: machine ${mname} running >${MACHINE_STARTED_MAX_NOTIFY_S}s" "Machine ${mname} (${mid}) started at ${started_at}, running for ${uptime_s} seconds. Consider suspending to save costs."
        echo "${start_ts}" > "${notified_file}"
      fi
    done
  fi

  file_mtime=$(stat -c %Y /activity/most_recent_call.txt 2>/dev/null || echo 0)
  age=$((now - file_mtime))
  echo "most_recent_call: ${age} seconds ago"
  if [ "${age}" -le "${MOST_RECENT_CALL_MAX_AGE_S}" ]; then
    continue
  fi

  json=$(cat "${MACHINES_CACHE}" 2>/dev/null || echo "[]")
  machine_id=$(echo "${json}" | jq -r '.[0].id // empty')
  state=$(echo "${json}" | jq -r '.[0].state // empty')
  if [ -z "${machine_id}" ]; then
    continue
  fi
  if [ "${state}" = "suspended" ] || [ "${state}" = "suspending" ]; then
    echo "machine ${machine_id} already ${state}, skipping suspend"
    continue
  fi
  curl -sf -X POST -H "${AUTH}" "${API}/${machine_id}/suspend" || true
  touch "${JUST_SUSPENDED}"
  echo "Suspended machine ${machine_id} (idle ${age} s)"
done
