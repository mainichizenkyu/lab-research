# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:00:57 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt

size=40
Fth=1
trans=0.2
step = 20000
numbers = 100

def transformation(U):
    U=U+Fth-np.max(U)+0.0001
    M=0
    while np.max(U)>=Fth:
        M=M+np.sum(U>Fth)
        dU=np.where(U<=Fth,0,U)
        largeinds=(U>=Fth)
        dU=dU*trans
        U=U+np.roll(dU,1,axis=0)+np.roll(dU,-1,axis=0)+np.roll(dU,1,axis=1)+np.roll(dU,-1,axis=-1)
        U[largeinds]=0
    return U,M,dU,largeinds
eq=[]
M=[]
eqfreq=[]
for j in range(numbers):
    U = np.random.rand(size, size)
    for i in range(step):
        U,m,a,b=transformation(U)
        '''fig = plt.figure(figsize=(6,6))
        plt.imshow(U, cmap=plt.cm.Oranges)
        plt.colorbar()'''
        M.append(m)
        eq.append(U[6,6])

fig = plt.figure(figsize=(10,6))
plt.plot(eq)    
fig = plt.figure(figsize=(10,6))
plt.plot(M)
M=np.array(M)
for i in range(130):
    eqfreq.append(np.sum(M == i+1))

fig = plt.figure(figsize=(10,6))
plt.yscale("log")
plt.xscale("log")
plt.plot(eqfreq,".")    

                
            