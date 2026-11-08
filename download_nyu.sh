#!/usr/bin/env bash
# Safe single-process download of the NYU Depth V2 labeled dataset (exactly 2972037809 bytes).
# Usage:  bash download_nyu.sh [dest_dir] [--fresh]
#   dest_dir  default ~/nyu_data  (keep it OUTSIDE Desktop/Documents so iCloud sync cannot interfere)
#   --fresh   delete any existing file first (use once; if interrupted, re-run WITHOUT --fresh to resume)
set -u
DEST="${1:-$HOME/nyu_data}"
FILE="$DEST/nyu_depth_v2_labeled.mat"
URL="${NYU_URL:-http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat}"
EXPECTED="${NYU_EXPECTED:-2972037809}"
mkdir -p "$DEST"
if pgrep -x curl >/dev/null 2>&1; then echo "A curl process is already running. Stop it first:  pkill curl"; exit 1; fi
LOCK="$DEST/.download.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "Lock folder $LOCK exists (another run?). If nothing is downloading, delete it: rmdir \"$LOCK\""; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
[ "${2:-}" = "--fresh" ] && rm -f "$FILE"
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
size() { if [ -f "$FILE" ]; then wc -c < "$FILE" | tr -d ' '; else echo 0; fi; }
for attempt in $(seq 1 40); do
  s=$(size)
  if [ "$s" -eq "$EXPECTED" ]; then break; fi
  if [ "$s" -gt "$EXPECTED" ]; then echo "File is larger than expected ($s bytes): corrupted. Deleting and restarting."; rm -f "$FILE"; fi
  echo "Attempt $attempt: have $(size) of $EXPECTED bytes"
  $CAF curl -L -C - --retry 5 --retry-delay 10 --speed-limit 20000 --speed-time 60 -o "$FILE" "$URL"
  sleep 3
done
s=$(size)
if [ "$s" -ne "$EXPECTED" ]; then echo "Incomplete: $s of $EXPECTED bytes. Run this script again (without --fresh) to resume."; exit 2; fi
echo "Size is correct ($s bytes). Now verify the contents:  python check_nyu.py \"$FILE\""
