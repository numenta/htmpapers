# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Plot location module representations during narrowing."""

import argparse
import base64
import cStringIO
import json
import math
import os
import xml.etree.ElementTree as ET

import matplotlib.cm
import numpy as np
import PIL.Image

from htmresearch.algorithms.location_modules import (
  ThresholdedGaussian2DLocationModule)

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def insertDiscreteModules(parent, activeCells, locationModuleWidth, rhombusBase,
                          rhombusHeight, stroke="lightgray"):
  cellPhasesAxis = np.linspace(0., 1., locationModuleWidth, endpoint=False)
  cellPhases = np.array([
    np.repeat(cellPhasesAxis, locationModuleWidth),
    np.tile(cellPhasesAxis, locationModuleWidth)]) + (0.5/locationModuleWidth)

  r = (0.5 * rhombusBase / locationModuleWidth)

  # Inserting a 'g' isn't necessary, but it makes this image more organized when
  # working with Illustrator layers.
  g = ET.SubElement(parent, "g")

  for cell, (phi1, phi2) in enumerate(cellPhases.T):
    circle = ET.SubElement(g, "circle")
    circle.set("cx", str(rhombusBase*(phi1 + phi2*np.cos(np.radians(60.)))))
    circle.set("cy", str(rhombusHeight*(1. - phi2)))
    circle.set("r", str(r))
    circle.set("stroke", stroke)
    circle.set("stroke-width", "1")

    if cell in activeCells:
      circle.set("fill", "black")
    else:
      circle.set("fill", "none")


def insertPointExcitations(parent, bumps, rhombusBase, rhombusHeight,
                           bumpSigma, opacity=0.6, enhancementFactor=4.0,
                           bumpOverlapMethod="probabilistic"):
  imgWidth = rhombusBase * 1.5
  imgHeight = rhombusBase * np.sin(np.radians(60.))

  numCols = int(rhombusBase * enhancementFactor)
  numRows = int(rhombusBase * np.sin(np.radians(60.)) * enhancementFactor)

  numBitmapRows = int(numRows)
  numBitmapCols = int(numCols * 1.5)

  # Look up the rows in bitmap order.
  queryPhases = np.array(
    [np.tile(np.linspace(0., 1., numCols, endpoint=False), numRows),
     np.repeat(np.linspace(0., 1., numRows, endpoint=False)[::-1],
               numCols)]) + [[0.5 / numCols],
                             [0.5 / numRows]]

  excitations = ThresholdedGaussian2DLocationModule.getCellExcitations(
    queryPhases, bumps, bumpSigma, bumpOverlapMethod)
  m = matplotlib.cm.get_cmap("rainbow")
  coloredSquare = m(excitations)
  coloredSquare[:,3] = opacity

  bitmap = np.zeros((numBitmapRows, numBitmapCols, 4))
  # Create a mapping from (row, columnInSquare) => columnInRhombus.
  # These rows are in bitmap order, starting from the top.
  columnOffsetByRow = np.floor(
    np.linspace(0, numCols * math.cos(np.radians(60.)), numRows, endpoint=False)
  )[::-1].astype("int")
  columnInRhombus = columnOffsetByRow[:, np.newaxis] + np.arange(numCols)
  bitmap[
    (np.repeat(np.arange(numRows), numCols),
     columnInRhombus.flatten())] = coloredSquare

  png = PIL.Image.fromarray(((bitmap) * 255).astype("uint8"), mode="RGBA")
  pngBuffer = cStringIO.StringIO()
  png.save(pngBuffer, format="PNG")
  pngStr = base64.b64encode(pngBuffer.getvalue())

  image = ET.SubElement(parent, "image")
  image.set("xlink:href", "data:image/png;base64,{}".format(pngStr))
  image.set("width", str(imgWidth))
  image.set("height", str(imgHeight))


def rhombusChart(inFilename, outFilename, objectNumber, moduleNumbers, numSteps,
                 rhombusBase=47, betweenX=4, betweenY=6):

  rhombusHeight = rhombusBase * np.sin(np.radians(60.))

  with open(inFilename, "r") as f:
    experiments = json.load(f)

  exp = next(exp for exp in experiments
             if exp[0]["numObjects"] == 50)

  locationModuleWidth = exp[0]["locationModuleWidth"]
  bumpSigma = exp[1]["bumpSigma"]

  locationLayerTimeline = exp[1]["locationLayerTimelineByObject"][str(objectNumber)]
  if numSteps is not None:
    locationLayerTimeline = locationLayerTimeline[:numSteps]

  # The final rhombus sticks out an additional 0.5 widths, hence the 0.5.
  width = (rhombusBase*(len(locationLayerTimeline) + 0.5) +
           betweenY*(len(locationLayerTimeline) - 1))
  height = (rhombusHeight*len(moduleNumbers) +
            betweenY*(len(moduleNumbers) - 1))
  svg = ET.Element("svg")
  svg.set("width", str(width + 20))
  svg.set("height", str(height + 20))
  svg.set("xmlns", "http://www.w3.org/2000/svg")
  svg.set("xmlns:xlink", "http://www.w3.org/1999/xlink")

  container = ET.SubElement(svg, "g")
  container.set("transform", "translate(10,10)")

  for t, moduleStates in enumerate(locationLayerTimeline):
    for iModule, moduleNumber in enumerate(moduleNumbers):
      moduleState = moduleStates[moduleNumber]

      rhombus = ET.SubElement(container, "g")
      rhombus.set("transform", "translate({},{})".format(
        t*(rhombusBase + betweenX), iModule*(rhombusHeight + betweenY)))

      insertDiscreteModules(rhombus, moduleState["activeCells"],
                            locationModuleWidth, rhombusBase, rhombusHeight)

      insertPointExcitations(rhombus, np.array(moduleState["bumps"]).T,
                             rhombusBase, rhombusHeight, bumpSigma)

  filename = os.path.join(CHART_DIR, outFilename)
  with open(filename, "w") as f:
    print "Saving", filename
    ET.ElementTree(svg).write(f, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile", type=str, required=True)
  parser.add_argument("--objectNumber", type=int, default=47)
  parser.add_argument("--moduleNumbers", type=int, nargs="+",
                      default=range(0,2,9))
  parser.add_argument("--numSteps", type=int, default=None)
  args = parser.parse_args()

  rhombusChart(args.inFile, args.outFile, args.objectNumber, args.moduleNumbers,
               args.numSteps)
