# ----------------------------------------------------------------------
# Copyright (C) 2017, Numenta Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc.  No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------
import os

from nab.plot import PlotNAB, MARKERS, getCSVData
from plotly.graph_objs import Data, Figure

dataFiles = (
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
)
dataNames = (
    "CPU Utilization (Percent)",
    "CPU Utilization (Percent)",
)
detectors=["numenta", "relativeEntropy", "twitterADVec", "skyline", "windowedGaussian", "bayesChangePt", "expose"]



def plotMultipleDetectors(plot,
                          resultsPaths,
                          detectors,
                          scoreProfile):

  traces = []
  traces.append(plot._addValues(plot.rawData))

  # Anomaly detections traces:
  for i, d in enumerate(detectors):
    threshold = plot.thresholds[d][scoreProfile]["threshold"]
    resultsData = getCSVData(os.path.join(plot.resultsDir, resultsPaths[i]))
    FP, TP = plot._parseDetections(resultsData, threshold)
    fpTrace, tpTrace = plot._addDetections("Detection by " + d,
                                           MARKERS[i], FP, TP)
    traces.append(fpTrace)
    traces.append(tpTrace)

  traces.append(plot._addWindows())
  traces.append(plot._addProbation())

  # Create plotly Data and Layout objects:
  data = Data(traces)
  layout = plot._createLayout("Anomaly Detections for " + plot.dataName)

  # Query plotly
  fig = Figure(data=data, layout=layout)
  plot_url = plot.py.plot(fig)

  return plot_url



# Create the list of result filenames for each detector
allResultsFiles = []
for f in dataFiles:
  resultFiles = []
  for d in detectors:
    filename = d + "/"+f.replace("/","/"+d+"_")
    resultFiles.append(filename)
  allResultsFiles.append(resultFiles)

# Now plot everything
for i in range(len(dataFiles)):
  dataPlotter = PlotNAB(
      dataFile=dataFiles[i],
      dataName=dataNames[i],
      offline=True)


  plotMultipleDetectors(
      dataPlotter,
      allResultsFiles[i],
      detectors=detectors,
      scoreProfile="standard")
