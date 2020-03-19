# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:30:02 2019

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

#ローレンツ方程式の定義
#ローレンツ方程式の定義
b=8/3
sigma = 10
gamma = 28

plt.rcParams["font.size"] = 30

def functionlorenz(x):
    return np.array((sigma*x[1]-sigma*x[0],x[0]*gamma-x[0]*x[2]-x[1],x[0]*x[1]-b*x[2]),dtype=float)
#周波数解析の関数の定義
def fft(x,t,n):
    plt.rcParams["font.size"] = 30
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
    i=fftfreq>0
    plt.figure(figsize=(12,6))
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,30)
    plt.xlabel('Frequency')
    plt.ylabel('PSD(dB) of $x^{(5)}$')
    
#ローレンツ方程式からデータの取得
h=0.001
rungekutta=RungeKutta1(functionlorenz)
x,y=rungekutta.calculation([10,10,10],h,200000)
plt.plot(x[:,0])

#HAVOKによる結果の取得
r=15
cal=Havok(x[:,0])
havok1,A1,B1,v1,dv1,vdf,V,S,U=cal.calculation2(100,r,r,h)


#周波数分析
t=np.linspace(0,199.900,199900) 

for i in range(6):
    fft(V[i, :], t, i+1)


#高階微分の算出
N = 15#微分階数をNとする. 



dc=[]


dx=[x[:,0]]
dy=[x[:,1]]
dz=[x[:,2]]
for i in range(15):
    dx.append(10*(dy[i]-dx[i]))
    com=[]
    for j in range(i+1):
        com.append(comb(i,j,exact=True)*dx[j]*dz[i-j])
    com=np.array(com).T
    com=np.sum(com,axis=1)
    dy.append(28*dx[i]-dy[i]-com)
    com1=[]
    for j in range(i+1):
        com1.append(comb(i,j,exact=True)*dx[j]*dy[i-j])
    com1=np.array(com1).T
    com1=np.sum(com1,axis=1)
    dz.append(com1-b*dz[i])

t=np.linspace(0,200.001,200001)    
for i in range(6):
    fft(dx[i]/np.std(dx[i])/np.sqrt(200000), t, i)
    
dx = np.array(dx)