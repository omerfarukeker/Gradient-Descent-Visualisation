# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 08:42:27 2019
LINEAR FITTER (GD & OLS)
@author: omer.eker
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("darkgrid")

#Ordinary Least Squares is an analytic model to fit the best line for linear data
def OLS(X,y):
    X_b = np.c_[np.ones(len(X)),X]
    thetas = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return thetas

#Gradient descent is a numeric optimisation algorithm to fit a best line for linear data.
#Following function returns the parameters and the dynamically plots the line fitting
def GD_plot(X,y,max_iter=100000,lrate=0.01):
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,projection='3d')
    fig.subplots_adjust(top=0.82)
    
    theta = np.random.randn(2)
    m = len(X)
    
    df = pd.DataFrame(np.zeros((len(X),3)),columns=["Theta0","Theta1","Error"])
    count = 0
    prediction = X*theta[1]+theta[0] 
    error = prediction-y
    MSE_temp = mean_squared_error(y,prediction)
    change_rate = 9999
    while change_rate > 0.001 and count < max_iter:
        theta[0] = theta[0] - (2/m)*lrate*sum(error) #last bit is the partial derivative of the loss function with respect to theta0
        theta[1] = theta[1] - (2/m)*lrate*sum(error*X) #last bit is the partial derivative of the loss function with respect to theta1
        prediction = X*theta[1]+theta[0]
        
        error = prediction-y
        MSE = mean_squared_error(y,prediction)
        change_rate = abs(MSE-MSE_temp)/MSE
        MSE_temp = MSE

        df.loc[count] = [theta[0],theta[1],MSE]
        
        #dynamic plotting of the line fitting process
        ax1.cla()
        ax1.scatter(X,y,label="Data")
        ax1.plot([min(X),max(X)],[min(X)*theta[1]+theta[0],max(X)*theta[1]+theta[0]],"r",linewidth=3,label=r"$y=\theta_1X+\theta_0$")
        ax1.legend(fancybox=True,loc="lower right")
        space_x = max(X)*0.2
        space_y = max(y)*0.4
        ax1.set_xlim([min(X)-space_x,max(X)+space_x])
        ax1.set_ylim([min(y)-space_y,max(y)+space_y])
        ax1.set_xlabel("X")
        ax1.set_ylabel("y")
        ax1.set_title("Gradient Descent",fontweight="bold")
        rsym0 = r'$\theta_0$'
        rsym1 = r'$\theta_1$'
        ax1.text(0,0.92,
                 f"{rsym0}={round(theta[0],2)}\n{rsym1}={round(theta[1],2)}\nMSE={round(MSE,2)}\nMSE Change Rate={round(change_rate,3)}",
                 ha="left",va="center",
                 transform = ax1.transAxes)
#        ax1.text(min(X)-space_x*0.8,max(y)+space_y*0.2,
#                 f"{rsym0}={round(theta[0],2)}\n{rsym1}={round(theta[1],2)}\nMSE={round(MSE,2)}\nMSE Change Rate={round(change_rate,3)}")
#        ax1.text(min(X)+space_x*2,max(y)+space_y*0.2,
#                 f"{rsym0}=\n{rsym1}=\nMSE=\nMSE Change Rate=",ha="right",va="center")
#        ax1.text(min(X)+space_x*2,max(y)+space_y*0.2,
#                 f"{round(theta[0],2)}\n{round(theta[1],2)}\n{round(MSE,2)}\n{round(change_rate,3)}",ha="left",va="center")
#        ax1.text(min(X)-space_x*0.8,max(y)+space_y*0.2,
#                 "{0:>20} = {1:>6}\n{2:>20} = {3:>6}\n{4:>20} = {5:>6}\n{6:>20} = {7:>6}"
#                 .format("t0",round(theta[0],2),"t1",round(theta[1],2),
#                         "MSE",round(MSE,2),"MSE Change Rate",round(change_rate,3)))
#        ax1.table(cellText=[[round(theta[0],2)],[round(theta[1],2)],[round(MSE,2)],[round(change_rate,3)]],
#                  rowLabels=[rsym0,rsym1,"MSE","MSE Change Rate"],
#                  loc="top left"
#                  )
        
        #dynamic plotting of the training error wrt theta values
        ax2.scatter(df.loc[count].Theta0,df.loc[count].Theta1,df.loc[count].Error,color="orange")
        ax2.set_xlabel(r"$\theta_0$",fontweight="bold")
        ax2.set_ylabel(r"$\theta_1$",fontweight="bold")
        ax2.set_zlabel("MSE",fontweight="bold")
        
        plt.pause(0.01)
        
        count += 1
        
    return theta,df

#Returns the GD line fitting animation
def GD_animation(X,y,max_iter=100000,lrate=0.01):
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,projection='3d')
    fig.subplots_adjust(top=0.82)
    
    camera = Camera(fig)
    theta = np.random.randn(2)
    m = len(X)
    
    df = pd.DataFrame(np.zeros((len(X),3)),columns=["Theta0","Theta1","Error"])
    count = 0
    prediction = X*theta[1]+theta[0] 
    error = prediction-y
    MSE_temp = mean_squared_error(y,prediction)
    change_rate = 9999
    
    space_x = max(X)*0.2
    space_y = max(y)*0.4
    rsym0 = r'$\theta_0$'
    rsym1 = r'$\theta_1$'
    
    while change_rate > 0.001 and count < max_iter:
        theta[0] = theta[0] - (2/m)*lrate*sum(error) #last bit is the partial derivative of the loss function with respect to theta0
        theta[1] = theta[1] - (2/m)*lrate*sum(error*X) #last bit is the partial derivative of the loss function with respect to theta1
        prediction = X*theta[1]+theta[0]
        
        error = prediction-y
        MSE = mean_squared_error(y,prediction)
        change_rate = abs(MSE-MSE_temp)/MSE
        MSE_temp = MSE
    
        df.loc[count] = [theta[0],theta[1],MSE]
        
        #dynamic plotting of the line fitting process
        ax1.scatter(X,y,color="deepskyblue")
        line = ax1.plot([min(X),max(X)],[min(X)*theta[1]+theta[0],max(X)*theta[1]+theta[0]],"r",linewidth=3)

        ax1.text(1,0,
                 f"{rsym0}={round(theta[0],2)}\n{rsym1}={round(theta[1],2)}\nMSE={round(MSE,2)}\nMSE Change Rate={round(change_rate,3)}",
                 ha="right",va="bottom",
                 transform = ax1.transAxes)
#        ax1.text(min(X)-space_x*0.8,max(y)+space_y*0.2,
#                 f"{rsym0}={round(theta[0],2)}\n{rsym1}={round(theta[1],2)}\nMSE={round(MSE,2)}\nMSE Change Rate={round(change_rate,3)}")

        ax1.set_xlim([min(X)-space_x,max(X)+space_x])
        ax1.set_ylim([min(y)-space_y,max(y)+space_y])
        ax1.set_xlabel("X")
        ax1.set_ylabel("y")
        ax1.legend(line,[r"$y=\theta_1X+\theta_0$"],loc="upper left")
#        ax1.set_title("Gradient Descent",fontweight="bold")
        ax2.set_xlabel(r"$\theta_0$",fontweight="bold")
        ax2.set_ylabel(r"$\theta_1$",fontweight="bold")
        ax2.set_zlabel("MSE",fontweight="bold")
        #dynamic plotting of the training error wrt theta values
        ax2.scatter(df.loc[:count].Theta0,df.loc[:count].Theta1,df.loc[:count].Error,color="deepskyblue")

        camera.snap()
        
        count += 1
    
    return theta,df,camera