#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_summary.py --bumpType gaussian --resultName results/convergenceSummary_gaussian.json --repeat 1

python plot_summary.py --inFile results/convergenceSummary_gaussian.json --outFile summary_gaussian.pdf --squeezeLegend
