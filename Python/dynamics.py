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
#from PIL import Image
from random import random
import elastica as el
import elastica_neurons as en
from holoviews import RGB
import inspect







def mises(a,k,ref,x):
    ''' basic von mises function
    Inputs
    -----------------------
    - a: determines magnitude
    - k: determines width (low k is wider width)
    - ref: reference angle
    - x: input angle
    '''
    return a*exp(k*(cos(2*(ref-x))))

'''
def mises_curve(a,k,angle):
    points = [i for i in np.arange(0, pi, 0.1)]
    points = [(points[i], mises(k,A,points[i],angle)) for i in range(len(points))]
    curve = hv.Curve(points)
    return curve
'''

def mises_curve(a,k,angle,neuron=32):
    points = [i for i in np.arange(0, pi, pi/(neuron+1))]
    points_mises = [ mises(k,a,points[i],angle) for i in range(len(points))]
    curve = hv.Curve(zip(points,points_mises))
    return curve


def plotbar(x,y,th,color='k',width=2,l=1, aspect=1):
    ''' 
    Plot a single bar 
    x,y: location middle of bar
    th(eta): orientation
    color: color (default = black)
    width: linewidth (default = 2)
    l:     line length (default = 1)
    Returns a holoviews curve object
    '''
    th = np.asarray(th)
    th = th + np.pi/2 # so that the orientation is relative to the vertical
    hl = l/2 # half length bar
    
    # define x and y points of bar
    X = [x-pl.sin(th)*hl,x+pl.sin(th)*hl]
    Y = [y-pl.cos(th)*hl,y+pl.cos(th)*hl]
    
    # return holoviews curve
    curve = hv.Curve(zip(X,Y))
    #return curve(plot={'xticks':x, 'yticks':y, 'zticks':l, 'aspect':aspect})(style={'alpha':0.4, 'linewidth':width})    
    return curve(plot={'aspect':aspect})(style={'alpha':0.4, 'linewidth':width})    



def printMatrix(testMatrix):
    print (' ')
    for i in range(int(np.sqrt(size(testMatrix)))):  # Make it work with non square matrices.
        print (i,)
    

def vonMises(a,k,angle,neuron):
    points_mises = np.array([ mises(k,a,neuron[i],angle) for i in range(len(neuron))])
    return points_mises

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def plotOneField(k,t,nosn,rs):
    result = hv.Points(zip(np.tile(t,nosn),rs[0,k[0],k[1],:].reshape(-1)))
    for j in range(1,rs.shape[0]):
        result *= hv.Points(zip(np.tile(t,nosn),rs[j,k[0],k[1],:].reshape(-1)))
    return result
    #return hv.Curve(zip(t,rs[:,1,1,:]))

def plotOneField2(orientations, k,t,nosn,rs):
    magnitude = np.empty(orientations.shape)
    for i in range(orientations.shape[0]):
        magnitude[i] = max(rs[i,k[0],k[1],:])
    size = m.sqrt(len(orientations))
    result = visualField3(orientations.reshape(size,size), magnitude.reshape(size,size))
    return result
    #return hv.Curve(zip(t,rs[:,1,1,:]))

def setNumberOfColors(num_colors):
    import colorsys
    colors = []
    for i in range(num_colors):
        (r,g,b) = colorsys.hsv_to_rgb(i/float(num_colors), 1.0, 0.75)
        color = '#%02x%02x%02x' % (int(r*256), int(g*256), int(b*256))
        colors.append(color)
    while (len(hv.core.options.Cycle().values)>0):
        del hv.core.options.Cycle().values[-1]
    for i in range(num_colors):
        hv.core.options.Cycle().values.append(colors[i])


def resetColors():
    while (len(hv.core.options.Cycle().values)>0):
        del hv.core.options.Cycle().values[-1]
    hv.core.options.Cycle().values.append('#30A2DA')
    hv.core.options.Cycle().values.append('#FC4F30')
    hv.core.options.Cycle().values.append('#E5AE38')
    hv.core.options.Cycle().values.append('#6D904F')
    hv.core.options.Cycle().values.append('#8B8B8B')


def oneColor():
    while (len(hv.core.options.Cycle().values)>0):
        del hv.core.options.Cycle().values[-1]
    hv.core.options.Cycle().values.append('#30A2DA')




def populationVector(orientations, r, nosn, timesteps):
    m = r.shape[1]
    n = r.shape[2]
    orientations = 2 * orientations
    sin_o = np.sin(orientations)
    cos_o = np.cos(orientations)
    r = np.reshape(r, (nosn, m*n, timesteps))
    r_x = np.zeros((m*n, timesteps))
    r_y = np.zeros((m*n, timesteps))
    for i in range(m*n):
        r_x[i,:] = np.dot(sin_o, np.squeeze(r[:,i,:]))
        r_y[i,:] = np.dot(cos_o, np.squeeze(r[:,i,:]))
    magnitude = np.sqrt(np.power(r_x,2) + np.power(r_y,2))
    r_x = np.divide(r_x, magnitude)
    r_y = np.divide(r_y, magnitude)
    direction = (np.arcsin(r_x)>0)*np.arccos(r_y) + (np.arcsin(r_x)<0)*(2*np.pi-np.arccos(r_y))
    direction = direction.real/2
    direction = np.reshape(direction, (m, n, timesteps))
    magnitude = np.reshape(magnitude, (m, n, timesteps))
    return (direction, magnitude)



# Visualization of presented stimulus

def visualField(orientations, aspect=1):
    renderer = hv.Store.renderers['matplotlib'].instance(fig='svg', holomap='gif')
    if len(orientations.shape)==2 :
        x_ind = []
        y_ind = []
        theta = []
        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):
                x_ind.append(i)
                y_ind.append(j)
                theta.append(orientations[i,j])
                #y_ind.append(orientations.shape[1]-1-j)
                #theta.append(orientations[j,i])
        bars = plotbar(x_ind, y_ind, theta, l=0.9, width=7, aspect=aspect)
    else:
        bars = plotbar(0,0,orientations,l=0.9, width=7, aspect=aspect)
    return bars


# Visualization of orientation selective neurons

def visualFieldColors(orientations):
    renderer = hv.Store.renderers['matplotlib'].instance(fig='svg', holomap='gif')
    if len(orientations.shape)==2:
        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):
                if (i==0 and j==0):
                    bars = plotbar(i,j,orientations[0,0], l=0.9, width=7)
                else:
                    bars *= plotbar(i, j, orientations[i,j], l=0.9, width=7)
    

    renderer.save(bars, 'orientations')

    bars1 = SVG(filename='orientations.svg')
    return bars



# Visualization of decoded stimulus

def visualFieldMagnitude(orientations, magnitude, aspect=1):
    #oneColor()
    renderer = hv.Store.renderers['matplotlib'].instance(fig='png', holomap='gif')
    if len(orientations.shape)==2:
        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):
                if (i==0 and j==0):
                    bars = plotbar(i,j,orientations[i,j], l=0.9, width=magnitude[i,j]*3)
                else:
                    bars *= plotbar(i, j, orientations[i,j], l=0.9, width=magnitude[i,j]*3)
    #renderer.save(bars, 'orientations')

    #bars1 = SVG(filename='orientations.svg')
    renderer.save(bars, 'example_I')
    parrot = RGB.load_image('example_I.png', array=True)
    rgb_parrot = RGB(parrot)
    rgb_parrot
    return bars(plot={'xticks':[0,orientations.shape[0]-1], 'yticks':[0,orientations.shape[1]-1], 'zticks':0.9, 'aspect':aspect})



# Generate a weight matrix based on the difference in orientation and distance

def covarianceMatrix(rate_matrix, distance_factor, orientation_factor):
    vector_length = np.size(rate_matrix)
    matrix = np.zeros((vector_length, vector_length))
    for i in range(vector_length):
        for j in range(vector_length):
            # Calculate the coordinates of the two neurons (x,y,preferred_orientation)
            first_neuron = calculateCoordinates(i, rate_matrix.shape)
            second_neuron = calculateCoordinates(j, rate_matrix.shape)
            # If the neurons respond to the same part of the visual field, don't have any connection between them
            if (first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]):
                continue
            # Model their connection based on distance and difference in orientation between them
            distance = np.sqrt(np.power(first_neuron[0]-second_neuron[0],2)+np.power(first_neuron[1]-second_neuron[1],2))
            angle_diff = min(np.mod(first_neuron[2]-second_neuron[2],rate_matrix.shape[2]), 
                             np.mod(second_neuron[2]-first_neuron[2],rate_matrix.shape[2]))
            matrix[i,j] = min(distance_factor, distance_factor*(1/exp(distance)))+\
                          min(orientation_factor, orientation_factor*(1/exp(angle_diff)))
            matrix[j,i] = matrix[i,j]
    return matrix


# Generate a weight matrix based on the elastica principle

def elasticaMatrix(m, n, nosn):
    # replicate a vector with different orientation of length nosn to an m x n x nosn matrix
    orientations = np.arange(0, np.pi, np.pi/nosn)
    orientations2 = np.expand_dims(orientations, axis=1)
    orientations3 = np.expand_dims(orientations2, axis=2)
    orientations4 = np.tile(orientations3, (1, m, n))
    vector_length = np.size(orientations4.ravel())
    matrix = np.zeros((vector_length, vector_length))
    unrolled_vector = orientations4.ravel()
    for i in range(vector_length):
            for j in range(vector_length):
                # Calculate the coordinates of the two neurons (x,y,preferred_orientation)
                first_neuron = calculateCoordinatesZ(i, orientations4.shape)
                second_neuron = calculateCoordinatesZ(j, orientations4.shape)
                # If the neurons respond to the same part of the visual field, don't have any connection between them
                if (first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]):
                    continue
                # Model the connection of the neurons according to the elastica principle
                x = first_neuron[1]-second_neuron[1]
                y = first_neuron[2]-second_neuron[2]
                theta1 = orientations4[first_neuron[0],first_neuron[1],first_neuron[2]]
                theta2 = orientations4[second_neuron[0],second_neuron[1],second_neuron[2]]
                energy = en.E(theta1,theta2,[x,y])
                matrix[i,j] = energy
                matrix[j,i] = energy
    return matrix


def calculateCoordinates(index,size_matrix):
    z_size = np.fix(index/(size_matrix[0]*size_matrix[1]))
    remainder = np.mod(index,(size_matrix[0]*size_matrix[1]))
    y_size = np.fix(remainder/size_matrix[0])
    x_size = np.mod(remainder,size_matrix[0])
    return np.array((x_size,y_size,z_size))


def calculateCoordinatesZ(index,size_matrix):
    x_size = np.fix(index/(size_matrix[1]*size_matrix[2]))
    remainder = np.mod(index,(size_matrix[0]*size_matrix[1]))
    y_size = np.fix(remainder/size_matrix[2])
    z_size = np.mod(remainder,size_matrix[2])
    return np.array((x_size,y_size,z_size))
