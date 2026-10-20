#!/usr/bin/env bash
# Build report/report.pdf from report/report.md with the figures embedded.
# Needs pandoc (>= 3.1) and typst:  brew install pandoc typst
set -euo pipefail
cd "$(dirname "$0")"
rm -rf figures && mkdir figures && cp ../results/figures/*.png figures/
pandoc report.md -o report.pdf --pdf-engine=typst \
  -V papersize=us-letter -V margin.x=2cm -V margin.y=2cm -V fontsize=10pt \
  -V mainfont="Helvetica Neue" --resource-path=.
echo "wrote $(pwd)/report.pdf"
