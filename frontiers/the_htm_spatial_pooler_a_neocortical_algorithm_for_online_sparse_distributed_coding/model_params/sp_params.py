spParamNoBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (1024, 1),
  "potentialRadius": 1024,
  "potentialPct": 0.5,
  "globalInhibition": True,
  "localAreaDensity": .02,
  "numActiveColumnsPerInhArea": -1,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.004,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "boostStrength": 0.0,
  "seed": 1936
}


spParamWithBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (1024, 1),
  "potentialRadius": 1024,
  "potentialPct": 1.0,
  "globalInhibition": True,
  "localAreaDensity": .02,
  "numActiveColumnsPerInhArea": -1,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.02,
  "synPermActiveInc": 0.1,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.001,
  "dutyCyclePeriod": 1000,
  "boostStrength": 100.0,
  "seed": 1936
}


spParamTopologyWithBoostingCross = {
  "inputDimensions": (32, 32),
  "columnDimensions": (32, 32),
  "potentialRadius": 5,
  "potentialPct": 1.0,
  "globalInhibition": False,
  "localAreaDensity": .1,
  "numActiveColumnsPerInhArea": -1,
  "wrapAround": True,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.02,
  "synPermActiveInc": 0.1,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "boostStrength": 100.0,
  "seed": 1936
}



spParamTopologyWithBoosting = {
  "inputDimensions": (32, 32),
  "columnDimensions": (32, 32),
  "potentialRadius": 12,
  "potentialPct": 1.0,
  "globalInhibition": False,
  "localAreaDensity": .02,
  "numActiveColumnsPerInhArea": -1,
  "wrapAround": True,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.02,
  "synPermActiveInc": 0.1,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "boostStrength": 100.0,
  "seed": 1936
}


spParamTopologyNoBoosting = {
  "inputDimensions": (32, 32),
  "columnDimensions": (32, 32),
  "potentialRadius": 10,
  "potentialPct": 1.0,
  "globalInhibition": False,
  "localAreaDensity": .1,
  "numActiveColumnsPerInhArea": -1,
  "wrapAround": True,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.1,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "boostStrength": 0.0,
  "seed": 1936
}
