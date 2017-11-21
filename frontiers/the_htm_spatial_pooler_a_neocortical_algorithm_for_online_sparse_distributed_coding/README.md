Spatial Pooler Research
=======================

This repository contains research scripts for the HTM spatial pooler (SP).
The goal is to demonstrate various computational properties of SP with
artificial and real dataset.

# SP metrics

The SP performance is quantified using a set of metrics
* entropy, which measures efficient use of columns
* stability, the same inputs should map to the same outputs even with continuous learning
* input-output overlap, which measures the noise robustness of SP output
* classification accuracy, which measures whether the SP outputs are classifiable

# Datasets

A variety of input datasets were considered, such as
* random SDRs ('randomSDR')
* random SDRs with varying sparsity ('randomSDRVaryingSparsity')
* correlated SDRs from two input fields ('correlatedSDRPairs')
* random bars (horizontal or vertical) ('randomBarPairs')
* random crosses ('randomCross')

# Usage
To run the SP experiment (without topology), use

`python train_sp -d DATASET -b USEBOOSTING -c 1 --spatialImp 'cpp' --trackOverlapCurve 1 `



