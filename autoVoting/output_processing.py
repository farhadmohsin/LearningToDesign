import numpy as np
from numpy import array
#UF, UW, SWF, SWW, U
UF = [0.022,0.023,0.022,0.023,0.024,0.022,0.022,0.023,0.022,0.022,0.022,0.022,0.023,0.023,0.023,0.023,0.021,0.024,0.022,0.023]
UW = [array([0.062, 0.061, 0.061, 0.061, 0.062]), array([0.064, 0.063, 0.063, 0.063, 0.064]), array([0.062, 0.061, 0.061, 0.061, 0.061]), array([0.065, 0.063, 0.063, 0.063, 0.063]), array([0.066, 0.064, 0.066, 0.065, 0.066]), array([0.062, 0.06 , 0.06 , 0.061, 0.06 ]), array([0.065, 0.062, 0.062, 0.063, 0.062]), array([0.064, 0.061, 0.062, 0.061, 0.062]), array([0.064, 0.063, 0.063, 0.063, 0.064]), array([0.065, 0.063, 0.064, 0.064, 0.064]), array([0.063, 0.062, 0.063, 0.063, 0.063]), array([0.064, 0.061, 0.062, 0.061, 0.062]), array([0.062, 0.06 , 0.061, 0.06 , 0.061]), array([0.065, 0.063, 0.063, 0.063, 0.064]), array([0.064, 0.063, 0.063, 0.063, 0.064]), array([0.065, 0.063, 0.063, 0.064, 0.064]), array([0.064, 0.062, 0.062, 0.062, 0.062]), array([0.064, 0.06 , 0.06 , 0.061, 0.061]), array([0.065, 0.064, 0.066, 0.064, 0.064]), array([0.063, 0.061, 0.06 , 0.061, 0.062])]
SWF = [512.157,511.439,511.054,511.313,511.005,509.073,510.305,510.908,511.342,511.857,510.762,510.784,512.541,510.942,510.883,512.39,510.515,511.575,511.139,511.685]
SWW = [array([529.393, 534.6  , 533.239, 533.412, 532.196]), array([529.461, 534.593, 533.287, 533.497, 532.471]), array([529.653, 534.708, 533.134, 533.439, 532.264]), array([529.888, 534.736, 533.127, 533.4  , 532.011]), array([528.943, 534.258, 532.815, 533.063, 531.72 ]), array([528.899, 533.956, 532.515, 532.716, 531.879]), array([529.519, 534.287, 532.56 , 532.989, 531.729]), array([529.571, 534.832, 533.362, 533.588, 532.469]), array([529.425, 535.107, 533.518, 533.745, 532.729]), array([529.059, 534.253, 532.754, 533.06 , 532.067]), array([529.831, 534.965, 533.679, 533.893, 532.761]), array([529.134, 534.231, 532.745, 533.105, 532.071]), array([528.691, 534.307, 532.701, 533.053, 532.105]), array([529.243, 534.551, 533.368, 533.444, 532.461]), array([529.604, 534.633, 533.105, 533.308, 532.172]), array([529.192, 534.808, 533.347, 533.726, 532.516]), array([529.448, 534.828, 533.438, 533.651, 532.141]), array([529.36 , 534.549, 533.199, 533.366, 532.438]), array([528.961, 534.73 , 533.414, 533.693, 532.46 ]), array([529.274, 534.756, 533.44 , 533.726, 532.53 ])]
U = [[534.6,   485.309],
 [534.593, 485.5  ],
 [534.708, 485.435],
 [534.736, 485.291],
 [534.258, 485.451],
 [533.956, 485.75 ],
 [534.287, 485.666],
 [534.832, 485.476],
 [535.107, 484.319],
 [534.253, 485.699],
 [534.965, 485.083],
 [534.231, 485.761],
 [534.307, 485.401],
 [534.551, 486.005],
 [534.633, 485.249],
 [534.808, 484.85 ],
 [534.828, 485.366],
 [534.549, 485.289],
 [534.73 , 485.121],
 [534.756, 485.154]]

print(np.mean(UF,axis=0))
print(np.mean(UW,axis=0))
print(np.mean(SWF,axis=0))
print(np.mean(SWW,axis=0))
print(np.mean(U,axis=0))