#!/bin/bash
for i in {1..10}
do
   python train_sp.py -t 1 -b 1 -d randomSDRVaryingSparsity --spatialImp cpp --trackOverlapCurve 1 -e 120 --changeDataSetAt 50 --name randomSDRVaryingSparsityContinuousLearning --seed $i &
done