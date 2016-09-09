import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets,tree
number_of_samples=100
x=np.linspace(-np.pi,np.pi,number_of_samples)
y=0.5*x+np.sin(x)+np.random.random(x.shape)
plt.scatter(x,y,color='black')
plt.xlabel('X-input feature')
plt.ylabel('Y-label values')
plt.title('Data for linear regression')
plt.show()
random_indices=np.random.permutation(number_of_samples)
x_train=x[random_indices[:70]]
y_train=y[random_indices[:70]]
x_val=x[random_indices[70:85]]
y_val=y[random_indices[70:85]]
x_test=x[random_indices[85:]]
x_test=y[random_indices[85:]]
model=linear_model.LinearRegression()
x_train_for_line_fitting=np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fitting=np.matrix(y_train.reshape(len(y_train),1))
model.fit(x_train_for_line_fitting,y_train_for_line_fitting)
plt.scatter(x_train,y_train,color='black')
plt.plot(x.reshape((len(x),1)),model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('X-input feature')
plt.ylabel('Y-target values')
plt.title('Fitting line')
plt.show()