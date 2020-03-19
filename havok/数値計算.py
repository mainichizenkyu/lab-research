# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:05:51 2019

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt

D = 30
H = 30
y = 30
def transformation(U):
    U1=(np.roll(U,1,axis=0)+np.roll(U,-1,axis=0)+np.roll(U,1,axis=1)+np.roll(U,-1,axis=-1))/4
    U1[:, 0] = 3*H
    U1[0, :] = y
    U1[:, -1] = 0
    U1[-1, :] = y
    U1[2*D+1:3*D+1,2*H+1:3*H+1]=0
    error = np.max((U-U1)**2)
    return U1, error
    
    
U = np.random.rand(5*D+2, 3*H+2)
error=[]
while(1):
    U, error_temp=transformation(U)
    error.append(error_temp)
    if error_temp < 10**(-6):
        break


fig = plt.figure(figsize=(8,6))
plt.imshow(np.transpose(U))
plt.colorbar()
    