# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:20:24 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1,RungeKutta3
import scipy as sp
import scipy.fftpack
def fft(x):
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./100)
    i=fftfreq>0
    plt.figure(figsize=(10,12))
    plt.subplot(211)
    plt.plot(x[0:50000])
    plt.xlabel('Time(s)')
    plt.subplot(212)
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,5)
    plt.ylim(-30,90)
    plt.xlabel('Frequency(1/s)')
    plt.ylabel('PSD(dB)')

data1 = np.loadtxt("bitcoin.csv",delimiter=",")
cal=Havok(data1*25)
q=100
havok1,A1,B1,v1,dv1,v15,V,S,U=cal.calculation2(q,4,4,0.01)
plt.figure(figsize=(12,6))
plt.plot(V[0,:],label="v1")
plt.plot(havok1[:,0],label="havok")
plt.legend()

for i in range(1):
    fft(V[i,:])
plt.figure(figsize=(15,9))
for i in range(10):
    plt.plot(U[:,i])
dc=[]
for i in range(V.shape[1]-2):
    dc.extend(np.transpose((V[:,i+1]-V[:,i])**2))
dc=np.reshape(dc,((V.shape[1])-2,-1))
print((sum(dc[:,0]))**0.5)   