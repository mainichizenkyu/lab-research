# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:23:00 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1,RungeKutta3
import scipy as sp
import scipy.fftpack
from scipy.special import perm, comb
b=8/3
def functionlorenz(x):
    return np.array((10*x[1]-10*x[0],x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-b*x[2]),dtype=float)
h=0.001
N=200000
#ローレンツ方程式からデータの取得
rungekutta=RungeKutta1(functionlorenz)
x,y=rungekutta.calculation([10,10,10],h,N)
x=np.delete(x,np.arange(0,10000),0)
y=np.delete(y,np.arange(0,10000),0)
plt.plot(x[:,0])
q=100
H=[]
for i in range(q):
    H.append(x[i:len(x[:,0])-1-q+i,0])
H=np.reshape(H,(q,-1))
SIGMA=1/(N-q)*np.dot(H,np.transpose(H))
la,S=np.linalg.eig(SIGMA)
y=np.dot(np.transpose(H),S)
Y=np.linalg.norm(y,axis=0)
y2=y/Y

cal=Havok(x[:,0])
m=7
y2df=y2[:,m-1]
y2df=np.delete(y2df,[0,1],0)
y1=np.delete(y2,range(m,y2.shape[1]),1)
print(np.shape(y1))
dy=cal.derivation5(np.transpose(y1),h)
y1=np.delete(np.transpose(y1),[0,1],1)
y1=np.delete(y1,range(30000,y1.shape[1]),1)
y1=np.transpose(y1)
dy=np.delete(dy,range(30000,dy.shape[0]),0)
print(np.shape(y1))
print(np.shape(dy))
A=cal.SINDy(y1, dy)
A2=np.delete(A,m-1,0)
A2=np.delete(A2,m-1,1)

B=np.delete(A,np.arange(0,m-1),1)
B=np.delete(B,m-1,0)


y0=y2[0,0:m-1]

havok=[y0]
for i in range(len(y2df)-1):
    y1=cal.calculate1(y0,h,y2df[i],y2df[i+1],B,A2,m)
    havok.append(y1)
    y0 =y1
havok=np.array(havok)
t=np.linspace(0,189.9,189900)
plt.figure(figsize=(12,6))
plt.plot(t[:10000],y2[:10000,0],label="v1")
plt.plot(t[:10000],havok[:10000,0],label="havok")