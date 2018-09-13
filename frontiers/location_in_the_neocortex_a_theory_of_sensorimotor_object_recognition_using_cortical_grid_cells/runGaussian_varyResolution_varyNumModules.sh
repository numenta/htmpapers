#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

repetitions=1

python gaussian_simulation.py --numObjects 10 20 30 40 50 60 70 80 90 100 110 --numUniqueFeatures 100 --inverseReadoutResolution 2 --numModules 6 12 18 --resultName results/gaussian_varyNumModules_100_feats_2_resolution.json --repeat $repetitions --appendResults

python gaussian_simulation.py --numObjects 10 20 30 40 50 60 70 80 90 100 110 120 130 --numUniqueFeatures 100 --inverseReadoutResolution 3 --numModules 6 12 18 --resultName results/gaussian_varyNumModules_100_feats_3_resolution.json --repeat $repetitions --appendResults

python gaussian_simulation.py --numObjects 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 --numUniqueFeatures 100 --inverseReadoutResolution 4 --numModules 6 12 18 --resultName results/gaussian_varyNumModules_100_feats_4_resolution.json --repeat $repetitions --appendResults
