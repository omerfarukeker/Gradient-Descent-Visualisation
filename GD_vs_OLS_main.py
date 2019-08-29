# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:37:17 2019
linear line fitting to a 2D data 
@author: omer.eker
"""

import numpy as np
import matplotlib.pyplot as plt
from GD_OLS_fitter import OLS,GD_plot,GD_animation 

#animation or regular plot
isanim = 1

#generate a 2D linear data with noise in the following format
# y = theta0+theta1*X
sample_size = 280
x_min,x_max = np.random.uniform(-10,-5),np.random.uniform(10,20)
X = np.random.uniform(x_min,x_max,sample_size)
theta0,theta1 = np.random.uniform(-300,300),np.random.uniform(2,10)
print(f"Theta0 = {theta0}\nTheta1 = {theta1}")
noise = np.random.normal(0,np.random.uniform(30,50),sample_size)
y_real = theta1*X+theta0+noise

#add outlier
#y_real[[35,210]] = [y_real.max()*5,-y_real.max()*5]

#%% call Ordinary Least Squares
thetas_ols = OLS(X,y_real)
newX = np.c_[np.ones((2,1)),[x_min,x_max]]
y_OLS = newX.dot(thetas_ols)

#%% call Gradient Descent (Batch)
if isanim:
    thetas_gd,df_gd,camera = GD_animation(X,y_real)
    anim = camera.animate()
    #use pillowWriters for saving gif files
#    anim.save("GD fit animation.gif",writer="pillow")
else:
    thetas_gd,df_gd,camera = GD_plot(X,y_real)

y_GD = newX.dot(thetas_gd)

#%% plot OLS and GD altogether
plt.figure()
dataplot = plt.scatter(X,y_real,label="Data")
olsplot = plt.plot(newX[:,1],y_OLS,color="sandybrown",linewidth=5,label="OLS")
gdplot = plt.plot(newX[:,1],y_GD,"deepskyblue",linewidth=5,label="GD")
plt.legend()
plt.ylabel("y")
plt.xlabel("X")
plt.title("GD vs OLS",fontweight="bold")

