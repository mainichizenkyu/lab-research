# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:20:07 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from rungekutta import RungeKutta3
from havok import Havok
import random
import scipy as sp
import scipy.fftpack

def function(x,a,t):
    return np.array((x[2],x[3],a[0]*(x[1]-x[0])-x[0]+a[1]*t-1/(1+np.abs(x[2])/a[2])*np.sign(x[2]),a[0]*(x[0]-x[1])-x[1]+a[1]*t-1/(1+np.abs(x[3])/a[3])*np.sign(x[3])),dtype=float)

def fft(x,t,n):
    x_fft=sp.fftpack.fft(x)
    x_psd=np.abs(x_fft)**2
    fftfreq = sp.fftpack.fftfreq(len(x_psd),1./1000)
    i=fftfreq>0
    plt.figure(figsize=(10,12))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlabel('Time(s)')
    n=n+1
    plt.ylabel('V'+str(n))
    plt.subplot(212)
    plt.plot(fftfreq[i],10*np.log10(x_psd[i]))
    plt.xlim(0,1)
    plt.ylim(-40,80)
    plt.xlabel('Frequency(1/s)')
    plt.ylabel('PSD(dB)')

def calculate(x,h,a,t):
    s1=function(x,a,t)
    s2=function(x+0.5*s1*h,a,t)
    s3=function(x+0.5*s2*h,a,t)
    s4=function(x+h*s3,a,t)
    return x+h/6*(s1+2*s2+2*s3+s4),h/6*(s1+2*s2+2*s3+s4)
    
def calculation(eq,a):
    step=0.001
    x0=[0.20+random.uniform(-0.1,0.1),0.2,0.3+random.uniform(-0.1,0.1),0.25]
    t=0
    n=0   
    sikiiti=0.1
    while n<=eq:
        x1,y1=calculate(x0,step,a,t)
        t=t+step
        if x1[2]>sikiiti and x0[2]<sikiiti and n<eq:
            n=n+1
            x0=x1
        elif x1[2]>sikiiti and x0[2]<sikiiti and n==eq:
            return t
            break
        x0=x1
if __name__ =='__main__':
    a=(1,0.1,1,1)
    number=20000
    numberofearthquake=5
    waitingtime=[]
    for i in range(number):
        print(i)
        t=calculation(numberofearthquake,a)
        waitingtime.append(t)
    plt.figure(figsize=(12,6))
    plt.hist(waitingtime,bins=20,rwidth=0.8,normed=True)
    plt.xlabel("witing time")
    plt.ylabel("distribution")
        
        


            
    
    