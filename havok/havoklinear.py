# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:25:00 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1,RungeKutta3
import scipy as sp
import scipy.fftpack
h=0.0001
def functionlorenz(x):
    return np.array((10*x[1]-10*x[0],x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-2.6667*x[2]),dtype=float)
def functionex1(x,a,t):
    return np.array((x[1],-0.3*x[0]-0.4*x[1]+0*(np.sin(2*np.pi*1*t)-1/2*np.sin(4*np.pi*1*t)+1/3*np.sin(6*np.pi*1*t)-1/4*np.sin(8*np.pi*1*t)+1/5*np.sin(10*np.pi*1*t)))+1*np.sqrt(h)*np.random.randn(),dtype=float)

def fft(x,t,n):
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./10000)
    i=fftfreq>0
    plt.figure(figsize=(10,12))
    plt.subplot(211)
    plt.plot(t[0:1000000],x[0:1000000])
    plt.xlim(0,100)
    plt.xlabel('Time(s)')
    n=n+1
    plt.ylabel('V'+str(n))
    plt.subplot(212)
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,10)
    plt.ylim(-30,90)
    plt.xlabel('Frequency(1/s)')
    plt.ylabel('PSD(dB)')

#ローレンツ方程式からデータの取得
rungekutta=RungeKutta3(functionex1)
x,y,t=rungekutta.calculation([0.01,0],h,2000000,0)
plt.plot(x[:,0])
x=np.delete(x,np.arange(0,10000),0)
y=np.delete(y,np.arange(0,10000),0)
cal=Havok(x[:,0])
havok1,A1,B1,v1,dv1,v15,V,S,U=cal.calculation2(200,3,3,h)
plt.rcParams["font.size"] = 14
t=np.linspace(0,189.9,1899000)
plt.figure(figsize=(12,6))
plt.plot(t[150000:175000],V[0,150000:175000],label="v1")
plt.plot(t[150000:175000],havok1[150000:175000,0],label="havok")
plt.xlabel("time(s)")
plt.ylabel("$v_1$")
plt.legend()
plt.figure(figsize=(18,9))
plt.plot(t[0:1750000],V[0,0:1750000],label="v1")
plt.plot(t[0:1750000],havok1[0:1750000,0],label="havok")
plt.legend()
for i in range(2):
    fft(V[i,:],t,i)
    
F=-0.3*x[0:189900,0]-0.4*x[0:189900,1]+np.sin(2*np.pi*1*t)-1/2*np.sin(4*np.pi*1*t)+1/3*np.sin(6*np.pi*1*t)-1/4*np.sin(8*np.pi*1*t)
plt.figure(figsize=(15,9))
for i in range(10):
    plt.plot(U[:,i])
dc=[]
for i in range(V.shape[1]-2):
    dc.extend(np.transpose((V[:,i+1]-V[:,i])**2))
dc=np.reshape(dc,((V.shape[1])-2,-1))
print((sum(dc[:,0]))**0.5)