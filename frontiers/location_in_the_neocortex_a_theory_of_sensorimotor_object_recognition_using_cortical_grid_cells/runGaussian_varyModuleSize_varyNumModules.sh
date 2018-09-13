#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

repetitions=1

python gaussian_simulation.py --numObjects 10 30 50 70 90 110 130 150 170 --numUniqueFeatures 100 --inverseReadoutResolution 3 --numModules 6 12 18 --enlargeModuleFactor 1.0 --resultName results/varyModuleSize_100_feats_1_enlarge.json --repeat $repetitions --appendResults

python gaussian_simulation.py --numObjects 10 30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 330 350 --numUniqueFeatures 100 --inverseReadoutResolution 3 --numModules 6 12 18 --enlargeModuleFactor 2.0 --resultName results/varyModuleSize_100_feats_2_enlarge.json --repeat $repetitions --appendResults

python gaussian_simulation.py --numObjects 50 100 150 200 250 300 350 400 450 500 550 600 --numUniqueFeatures 100 --inverseReadoutResolution 3 --numModules 6 12 18 --enlargeModuleFactor 3.0 --resultName results/varyModuleSize_100_feats_3_enlarge.json --repeat $repetitions --appendResults
