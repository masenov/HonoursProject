# Imports
import math as m
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
import pylab as pl
from pylab import exp,cos,sin,pi,tan, pi
import pandas as pd
import seaborn as sb
import holoviews as hv
from IPython.display import SVG
import io
from PIL import Image
from random import random
import elastica as el
import elastica_neurons as en
from dynamics import *
m = 1
n = 1
nosn = 9
distance_scaling = 0.00005
orientation_scaling = 0.00005
np.size(np.ones((m,n,nosn)))
a = covarianceMatrix(np.ones((m,n,nosn)),distance_scaling,orientation_scaling)
a.shape
plt.imshow(a)
from tempfile import TemporaryFile
matrix_file = TemporaryFile()
np.savez(matrix_file, a)
matrix_file.seek(0)