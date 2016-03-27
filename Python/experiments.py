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
import os.path
from dynamics import *




def norm_plot(values):
	values[values>2] = values[values>2] - np.pi
	values[values<-2] = values[values<-2] + np.pi
	return values


#----- Calculate matrix for fig. B-----
def calculateMatrixB(nosn=100):
	# used to calculate the matrix
	E0 = 4
	m = 1
	n = 3
	orientations = np.arange(0, np.pi, np.pi/nosn)
	orientations2 = np.expand_dims(orientations, axis=1)
	orientations3 = np.expand_dims(orientations2, axis=2)
	orientations4 = np.tile(orientations3, (1, m, n))
	orientations4 = np.swapaxes(orientations4,0,1)
	orientations4 = np.swapaxes(orientations4,1,2)
	vector_length = np.size(orientations4.ravel())
	matrix = np.zeros((vector_length, vector_length))
	for i in range(vector_length):
		for j in range(vector_length):
			# Calculate the coordinates of the two neurons (x,y,preferred_orientation)
			first_neuron = calculateCoordinatesNew(i, orientations4.shape)
			second_neuron = calculateCoordinatesNew(j, orientations4.shape)
			# If the neurons respond to the same part of the visual field, don't have any connection between them
			if (first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]):
				continue
			# Model the connection of the neurons according to the elastica principle
			y = first_neuron[0]-second_neuron[0]
			x = first_neuron[1]-second_neuron[1]
			theta1 = orientations4[first_neuron[0],first_neuron[1],first_neuron[2]]
			theta2 = orientations4[second_neuron[0],second_neuron[1],second_neuron[2]]
			energy = en.E(theta1,theta2,[x,y])
			distance = np.sqrt(np.power(x,2) + np.power(y,2))
			#matrix[i,j] = -(energy-E0)/distance
			matrix[i,j] = -(energy-E0)/distance
			matrix[j,i] = matrix[i,j]
	#showWeights(matrix, fig_size=10)
	return matrix



#----- Calculate matrix for fig. C -----
def calculateMatrixC(x,y,nosn=100):
	E0 = 4
	m = 1
	n = 3
	orientations = np.arange(0, np.pi, np.pi/nosn)
	orientations2 = np.expand_dims(orientations, axis=1)
	orientations3 = np.expand_dims(orientations2, axis=2)
	orientations4 = np.tile(orientations3, (1, m, n))
	orientations4 = np.swapaxes(orientations4,0,1)
	orientations4 = np.swapaxes(orientations4,1,2)
	vector_length = np.size(orientations4.ravel())
	matrix2 = np.zeros((vector_length, vector_length))
	for i in range(vector_length):
		for j in range(vector_length):
			# Calculate the coordinates of the two neurons (x,y,preferred_orientation)
			first_neuron = calculateCoordinatesNew(i, orientations4.shape)
			second_neuron = calculateCoordinatesNew(j, orientations4.shape)
			# If the neurons respond to the same part of the visual field, don't have any connection between them
			if ((first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]) or matrix2[i,j]!=0):
				continue
			# Model the connection of the neurons according to the elastica principle
			#y = first_neuron[0]-second_neuron[0]
			#x = first_neuron[1]-second_neuron[1]
			mult_y = np.max([np.abs(first_neuron[0]-second_neuron[0]),1])
			mult_x = np.max([np.abs(first_neuron[1]-second_neuron[1]),1])
			theta1 = orientations4[first_neuron[0],first_neuron[1],first_neuron[2]]
			theta2 = orientations4[second_neuron[0],second_neuron[1],second_neuron[2]]
			energy = en.E(theta1,theta2,[x*mult_x,y*mult_y])
			distance = np.sqrt(np.power(x,2) + np.power(y,2))
			matrix2[i,j] = -(energy-E0)
			matrix2[j,i] = matrix2[i,j]
	matrix2[0:nosn,2*nosn:3*nosn] /= 2
	matrix2[2*nosn:3*nosn,0:nosn] /= 2
	return matrix2
	#showWeights(matrix, fig_size=10)


#----- Calculate matrix for fig. D -----
def calculateMatrixD(nosn=100):
	E0 = 4
	m = 1
	n = 3
	orientations = np.arange(0, np.pi, np.pi/nosn)
	orientations2 = np.expand_dims(orientations, axis=1)
	orientations3 = np.expand_dims(orientations2, axis=2)
	orientations4 = np.tile(orientations3, (1, m, n))
	orientations4 = np.swapaxes(orientations4,0,1)
	orientations4 = np.swapaxes(orientations4,1,2)
	vector_length = np.size(orientations4.ravel())
	matrix = np.zeros((vector_length, vector_length))
	for i in range(vector_length):
		for j in range(vector_length):
			# Calculate the coordinates of the two neurons (x,y,preferred_orientation)
			first_neuron = calculateCoordinatesNew(i, orientations4.shape)
			second_neuron = calculateCoordinatesNew(j, orientations4.shape)
			# If the neurons respond to the same part of the visual field, don't have any connection between them
			if (first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]):
				continue
			# Model the connection of the neurons according to the elastica principle
			y = first_neuron[0]-second_neuron[0]
			x = first_neuron[1]-second_neuron[1]
			theta1 = orientations4[first_neuron[0],first_neuron[1],first_neuron[2]]
			theta2 = orientations4[second_neuron[0],second_neuron[1],second_neuron[2]]
			energy = en.E(theta1,theta2,[x,y])
			distance = np.sqrt(np.power(x,2) + np.power(y,2))
			matrix[i,j] = -(energy-E0)
			matrix[j,i] = matrix[i,j]
	#showWeights(matrix, fig_size=10)
	return matrix



#----- Calculate matrix for fig. F -----
def calculateMatrixF(locations, nosn=100):
	m = 1
	n = 7
	nn = nosn
	E0 = 4
	orientations = np.arange(0, np.pi, np.pi/nosn)
	orientations2 = np.expand_dims(orientations, axis=1)
	orientations3 = np.expand_dims(orientations2, axis=2)
	orientations4 = np.tile(orientations3, (1, m, n))
	orientations4 = np.swapaxes(orientations4,0,1)
	orientations4 = np.swapaxes(orientations4,1,2)
	vector_length = np.size(orientations4.ravel())
	matrix = np.zeros((vector_length, vector_length))
	for i in range(vector_length):
		for j in range(vector_length):
			# Calculate the coordinates of the two neurons (x,y,preferred_orientation)
			first_neuron = calculateCoordinatesNew(i, orientations4.shape)
			second_neuron = calculateCoordinatesNew(j, orientations4.shape)
			# If the neurons respond to the same part of the visual field, don't have any connection between them
			if (first_neuron[0]==second_neuron[0] and first_neuron[1]==second_neuron[1]):
				continue
			y = first_neuron[0]-second_neuron[0]
			y = locations[first_neuron[1],1]-locations[second_neuron[1],1]
			x = locations[first_neuron[1],0]-locations[second_neuron[1],0]
			# Model the connection of the neurons according to the elastica principle
			theta1 = orientations4[first_neuron[0],first_neuron[1],first_neuron[2]]
			theta2 = orientations4[second_neuron[0],second_neuron[1],second_neuron[2]]
			energy = en.E(theta1,theta2,[x,y])
			distance = np.sqrt(np.power(x,2) + np.power(y,2))
			matrix[i,j] = -(energy-E0)/distance
			matrix[j,i] = matrix[i,j]
	showWeights(matrix, fig_size=10)
	return matrix