#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_feature_distributions.py --bumpType gaussian --resultName results/featureDistributions_gaussian.json --repeat 1

python plot_feature_distributions.py --inFile results/featureDistributions_gaussian.json --outFile featureDistributions_gaussian.pdf --xlim2 30
