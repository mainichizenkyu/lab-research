# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:19:39 2018

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

data = np.loadtxt('U-2834a.txt', comments='SOUTH', delimiter=',', dtype='float')
data1=list(np.array(data[:,0]))
data1=np.array(data1*25)
cal=Havok(data1)
q=200
havok1,A1,B1,v1,dv1,v15,V,S,U=cal.calculation2(q,8,8,0.01)
t=np.linspace(0,2587.50,258750)
plt.figure(figsize=(12,6))
plt.plot(t[0:258750-q-1],V[0,:],label="v1")
plt.figure(figsize=(12,6))
plt.plot(t[0:258750-q-3],havok1[:,0],label="havok")
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