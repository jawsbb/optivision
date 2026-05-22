#!/usr/bin/env bash
# Download the example images used by the notebook and the example scripts.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/assets"
mkdir -p "$DEST"

BASE="https://storage.googleapis.com/com-roboflow-marketing/playground-examples"
IMAGES=(
  "pexels-vanessa-loring-5966631.jpg"
  "pexels-eyup-sayar-290427017-18373303.jpg"
  "pexels-mutecevvil-18013812.jpg"
  "pexels-shvets-production-7195054.jpg"
  "pexels-spencer-4353558.jpg"
  "top-shot-of-a-worker-scanning-boxes-using-a-bar-co-2026-01-11-09-59-09-utc.jpg"
  "warehouse-workers-inspecting-boxes-along-conveyor-2026-01-11-09-55-23-utc.jpg"
  "top-view-of-people-relaxing-in-the-pool-on-yellow-2026-03-24-21-54-59-utc.jpg"
  "aerial-drone-photograph-of-traffic-jam-in-metropol-2026-03-18-17-36-02-utc.jpg"
)

for img in "${IMAGES[@]}"; do
  echo "Downloading $img"
  curl -fsSL "$BASE/$img" -o "$DEST/$img"
done

echo "Done. ${#IMAGES[@]} images saved to $DEST"
