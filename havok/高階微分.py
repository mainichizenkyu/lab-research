# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:36:25 2019

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1, RungeKutta3
import scipy as sp
import scipy.fftpack
from scipy.special import perm, comb
from mpl_toolkits.mplot3d import Axes3D


def functionlorenz(x, sigma, b, gamma):
    return np.array((sigma*x[1]-sigma*x[0],x[0]*gamma-x[0]*x[2]-x[1],x[0]*x[1]-b*x[2]),dtype=float)

def calculatelorenz():
    h=0.001
    rungekutta=RungeKutta1(functionlorenz)
    x,y=rungekutta.calculation([10,10,10],h,200000)
    return x


def higherderivative():
    x = calculatelorenz()
    dx=[x[:,0]]
    dy=[x[:,1]]
    dz=[x[:,2]]
    for i in range(11):
        dx.append(10*(dy[i]-dx[i]))
        com=[]
        for j in range(i+1):
            com.append(comb(i,j,exact=True)*dx[j]*dz[i-j])
        com=np.array(com).T
        com=np.sum(com,axis=1)
    print(com)
    dy.append(28*dx[i]-dy[i]-com)
    com1=[]
    for j in range(i+1):
        com1.append(comb(i,j,exact=True)*dx[j]*dy[i-j])
    com1=np.array(com1).T
    com1=np.sum(com1,axis=1)
    dz.append(com1-b*dz[i])
