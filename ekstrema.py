from scipy import signal
import numpy as np


xs = np.arange(0, np.pi, 0.05)
data = np.sin(xs)
peakind = signal.find_peaks_cwt(data, np.arange(1,10))
peakind, xs[peakind], data[peakind]


#Calculate the relative maxima of data.
from scipy.signal import argrelmax
x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
argrelmax(x)

y = np.array([[1, 2, 1, 2],
              [2, 2, 0, 0],
              [5, 3, 4, 4]])

argrelmax(y, axis=1)


#Calculate the relative extrema of data.
from scipy.signal import argrelextrema
x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
argrelextrema(x, np.greater)

y = np.array([[1, 2, 1, 2],
              [2, 2, 0, 0],
              [5, 3, 4, 4]])

argrelextrema(y, np.less, axis=1)



