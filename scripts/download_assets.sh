#!/usr/bin/env bash
# Download the MediaPipe Face Landmarker model and 60 Free Spoken Digit Dataset recordings.
set -euo pipefail
mkdir -p models data/voices
wget -q -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
tmp="$(mktemp -d)"
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git "$tmp/fsdd"
cp "$tmp"/fsdd/recordings/[0-9]_*_0.wav data/voices/
rm -rf "$tmp"
echo "Model and voice samples downloaded"
