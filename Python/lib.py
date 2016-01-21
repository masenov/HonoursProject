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


def plotbar(x,y,th,color='k',width=2,l=1):
    ''' 
    Plot a single bar 
    x,y: location middle of bar
    th(eta): orientation
    color: color (default = black)
    width: linewidth (default = 2)
    l:     line length (default = 1)
    Returns a holoviews curve object
    '''
    #    th += pl.pi/4 # so that the orientation is relative to the vertical
    hl = l/2 # half length bar
    
    # define x and y points of bar
    X = [x-pl.sin(th)*hl,x+pl.sin(th)*hl]
    Y = [y-pl.cos(th)*hl,y+pl.cos(th)*hl]
    
    # return holoviews curve
    return hv.Curve(zip(X,Y), xaxis="")#(style={'alpha':0.4})    

def mises_curve(a,k,angle):
    points = [i for i in np.arange(0, pi, 0.1)]
    points = [(points[i], mises(k,A,points[i],angle)) for i in range(len(points))]
    curve = hv.Curve(points)
    return curve

def mises_curve(a,k,angle,neuron):
    points = [i for i in np.arange(0, pi, pi/(neuron+1))]
    points_mises = [ mises(k,A,points[i],angle) for i in range(len(points))]
    curve = hv.Curve(zip(points,points_mises))
    return curve

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

r = np.zeros((9,10,10))
r_pad = np.lib.pad(r, 1, padwithzeros)