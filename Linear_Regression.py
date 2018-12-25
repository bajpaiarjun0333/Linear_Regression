#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 21:59:33 2018

@author: bajpaiarjun0333
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Setting the seed for similar results
np.random.seed(101)
tf.set_random_seed(101)


#Generating linear data points randomly
#50 data points generated
x=np.linspace(0,50,50)
y=np.linspace(0,50,50)

#adding some noise to the data points
x+=np.random.uniform(-4,4,50)
y+=np.random.uniform(-4,4,50)

#find the length of the datapoints
n=len(x)

#visualizing the data points
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data Points")
plt.show()

#Create the placeholder to supply data
X=tf.placeholder("float")
Y=tf.placeholder("float")

#Defining the weight and bias variables
W=tf.Variable(np.random.randn(),name="Weight")
b=tf.Variable(np.random.randn(),name="Bias")


#definin the learning rate 
learning_rate=0.001
training_epochs=10000

#Building the hypothesis
y_pred=tf.add(tf.multiply(X,W),b)
#cost of the operation
cost=tf.reduce_sum(tf.pow((y_pred-Y),2))/(2*n)
#attaching the optimizer to the tensorflow graph
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#initializing the global variables
init=tf.global_variables_initializer()

#Starting the tensorflow session 
with tf.Session() as sess:
    
    sess.run(init)
    
    #running the epochs
    for epochs in range(training_epochs):
        for (_x,_y) in zip(x,y):
            
            sess.run(optimizer,feed_dict={X:_x,Y:_y})
            #displaying the result after 50 iterations
        if (epochs+1)%50==0:
                c=sess.run(cost,feed_dict={X:x,Y:y})
                print("Epoch:   ",(epochs+1),":Cost:   ",c)
    #saving the output for future prediction of the values
    training_cost=sess.run(cost,feed_dict={X:x,Y:y})
    weight=sess.run(W)
    bias=sess.run(b)


#ready to make prediction
prediction=weight*x+bias
print("Training cost:  ",training_cost,"Weight:    ",weight,"Bias:    ",bias)


#plotting the results
plt.plot(x,y,'ro',label="Original Data")
plt.plot(x,prediction,label="Fitted Line")
plt.title("Linear Regression Result")
plt.legend()
plt.show()
#The Most Close Fit Line Has Been Obtained

