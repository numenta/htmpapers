#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --numModules 10 --bumpType gaussian --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 26 30 40 --resultName results/comparisonToIdeal_gaussian.json --repeat 10

python ideal_sim.py &
python bof_sim.py &
wait

python plot_comparison_to_ideal.py --inFile results/comparisonToIdeal_gaussian.json --outFile comparisonToIdeal_gaussian.pdf --locationModuleWidth 26 30 40 --legendPosition 1.0 0.6
