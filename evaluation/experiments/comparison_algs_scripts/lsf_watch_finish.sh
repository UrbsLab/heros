#!/usr/bin/env bash
set -euo pipefail

# Watcher-only: submit one LSF job that waits until a given user has ZERO unfinished jobs,
# then triggers an LSF email notification on completion.
#
# Usage:
#   ./lsf_watch_user_idle.sh -e you@domain.com [-U user] [-q queue] [-P poll_seconds] [-J watcher_job_name]
#
# Notes:
# - This watches ALL unfinished jobs for the user (PEND/RUN/etc). Any new jobs the user submits will extend the wait.
# - Email content is whatever your LSF cluster sends for -N notifications.

EMAIL="raptor419heavy@gmail.com"
WATCH_USER="bandheyh"
QUEUE="i2c2_normal"
POLL_SECONDS=30
WATCHER_NAME="notify_user_idle"

while getopts ":e:U:q:P:J:" opt; do
  case "$opt" in
    e) EMAIL="$OPTARG" ;;
    U) WATCH_USER="$OPTARG" ;;
    q) QUEUE="$OPTARG" ;;
    P) POLL_SECONDS="$OPTARG" ;;
    J) WATCHER_NAME="$OPTARG" ;;
    *) echo "Usage: $0 -e you@domain.com [-U user] [-q queue] [-P poll_seconds] [-J watcher_job_name]" >&2; exit 2 ;;
  esac
done

if [[ -z "${EMAIL}" ]]; then
  echo "Missing -e email address." >&2
  echo "Usage: $0 -e you@domain.com [-U user] [-q queue] [-P poll_seconds] [-J watcher_job_name]" >&2
  exit 2
fi
if [[ -z "${WATCH_USER}" ]]; then
  echo "Could not determine user. Provide -U <user>." >&2
  exit 2
fi

command -v bsub >/dev/null 2>&1 || { echo "bsub not found in PATH (LSF env not loaded)." >&2; exit 2; }
command -v bjobs >/dev/null 2>&1 || { echo "bjobs not found in PATH (LSF env not loaded)." >&2; exit 2; }

watcher_cmd=$(cat <<'EOF'
set -euo pipefail
WATCH_USER="__WATCH_USER__"
POLL_SECONDS="__POLL_SECONDS__"

echo "LSF watcher started on $(hostname) at $(date)"
echo "Monitoring unfinished jobs for user: ${WATCH_USER}"
echo "Poll interval: ${POLL_SECONDS}s"
echo "Caveat: new jobs submitted by ${WATCH_USER} will extend the wait."

while true; do
  # bjobs shows unfinished jobs by default; -u restricts to the user.
  # We strip header and blank lines; if nothing remains, user is idle.
  lines="$(bjobs -u "${WATCH_USER}" 2>/dev/null | tail -n +2 || true)"
  lines="$(echo "${lines}" | sed '/^[[:space:]]*$/d' || true)"

  if [[ -z "${lines}" ]]; then
    echo "No unfinished jobs for ${WATCH_USER} at $(date). Exiting watcher."
    break
  fi

  count="$(echo "${lines}" | wc -l | tr -d ' ')"
  echo "Unfinished jobs for ${WATCH_USER}: ${count} (as of $(date))"
  sleep "${POLL_SECONDS}"
done
EOF
)

watcher_cmd="${watcher_cmd/__WATCH_USER__/${WATCH_USER}}"
watcher_cmd="${watcher_cmd/__POLL_SECONDS__/${POLL_SECONDS}}"

# Submit the watcher job; -N requests email notification; -u sets the recipient.
if [[ -n "${QUEUE}" ]]; then
  bsub -q "${QUEUE}" -J "${WATCHER_NAME}_${WATCH_USER}" -N -u "${EMAIL}" bash -lc "${watcher_cmd}"
else
  bsub -J "${WATCHER_NAME}_${WATCH_USER}" -N -u "${EMAIL}" bash -lc "${watcher_cmd}"
fi
