#!/usr/bin/env bash
set -euo pipefail
IMG="gcr.io/$(gcloud config get-value project)/player-sim:latest"

# Build artifacts for both modes
python training/prepare_features.py --mode offense || true
python training/prepare_features.py --mode defense || true

# Build container
gcloud builds submit --tag "$IMG" .

# Deploy to Cloud Run
SERVICE="player-sim"
REGION="us-central1"

gcloud run deploy "$SERVICE" \
  --image "$IMG" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars ARTIFACT_DIR=/app/artifacts

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "Deployed: $URL"