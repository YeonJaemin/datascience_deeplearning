import numpy as np
import tensorflow as tf

"""
phase1. assemble a graph
"""
#load input and output data from file
xy  = np.loadtxt('xor_dataset.txt', upgack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

#define placeholders for input and output

#define the weights and bias

#define hypotheses

#define loss function

#define optimizer

"""
phrase2. use a session to execute operations in the graph
"""
with tf.session() as sess:
	#init variables
	#wirte your own code...

	#test trained model

	#check accuracy