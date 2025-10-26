#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(gcloud config get-value project)
BUCKET=${1:-"YOUR_BUCKET"}

echo "Loading offense CSV from gs://$BUCKET/stats_offense.csv"
bq load --source_format=CSV --skip_leading_rows=1 \
  football.stats_offense gs://$BUCKET/stats_offense.csv

echo "Loading defense CSV from gs://$BUCKET/stats_defense.csv"
bq load --source_format=CSV --skip_leading_rows=1 \
  football.stats_defense gs://$BUCKET/stats_defense.csv