# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:19:45 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from rungekutta import RungeKutta3
from havok import Havok
import scipy as sp
import scipy.fftpack
a=(1,0.1,1,1)
def function(x,a,t):
    return np.array((x[2],x[3],a[0]*(x[1]-x[0])-x[0]+a[1]*t-1/(1+np.abs(x[2])/a[2])*np.sign(x[2])+0*np.sqrt(0.001)*np.random.randn(),a[0]*(x[0]-x[1])-x[1]+a[1]*t-1/(1+np.abs(x[3])/a[3])*np.sign(x[3])),dtype=float)
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
rungekutta=RungeKutta3(function)
q=600000
x,y,t=rungekutta.calculation([0.20,0.2,0.3,0.25],0.001,q,a)
x=np.delete(x,np.arange(0,10000),0)
y=np.delete(y,np.arange(0,10000),0)
t=np.delete(t,np.arange(0,10000),0)
plt.figure(figsize=(6,6))
plt.plot(x[:,0]-a[1]*t+a[2]/(a[2]+a[1]),x[:,2])
plt.xlabel('$U_1-U_1^e$')
plt.ylabel('$\dot{U_1}$')
plt.axis('scaled')
cal=Havok(x[:,2])
havok1,A1,B1,v1,dv1,v15,V,S,U=cal.calculation2(100,6,6,0.001)

plt.rcParams["font.size"] = 15

plt.figure(figsize=(12,6))
plt.plot(t[150000:175000],V[0,150000:175000],label="v1")
plt.plot(t[150000:175000],havok1[150000:175000,0],label="havok")
plt.xlabel("time(s)")
plt.ylabel("$v_1$")
plt.legend()
t=np.delete(t,np.arange(0,101),0)
for i in range(6):
    fft(V[i,:],t,i)

eq=[]
for i in range(q-10102):
    if V[0,i]>V[0,i+1] and V[0,i+1]<V[0,i+2] and V[0,i+1]<-0.0002:
        eq.append(V[0,i+1])
plt.figure(figsize=(8,6))
plt.hist(eq,bins=20,log=True,rwidth=0.8)
eq1=[]
for i in range(q-10102):
    if x[i,2]<x[i+1,2] and x[i+1,2]>x[i+2,2] and x[i+1,2]>0.15:
        eq1.append(x[i+1,2])
plt.figure(figsize=(8,6))
plt.hist(eq1,bins=20,log=True,rwidth=0.8)

V6=np.sort(np.abs(V[5,:]))
Vborder=V6[int(V6.size*0.95)]
inter=[]
none=[]
t1=[]
t2=[]
interval=600
ran=0.001
for i in range(int(V6.size)):
    if np.abs(V[5,i])>Vborder:
        inter.append(V[0,i])
        t1.append(t[i])
    else:
        none.append(V[0,i])
        t2.append(t[i])
inter=np.ravel(inter)
t1=np.ravel(t1)
none=np.ravel(none)
t2=np.ravel(t2)
plt.figure(figsize=(12,6))
plt.plot(t2,none,label="Foercin Inactive")
plt.plot(t1,inter,".",label="Forcing Active")
plt.xlabel("time(s)")
plt.ylabel("$V_1$")
plt.legend(loc="lower right",fontsize=15)

plt.figure(figsize=(12,6))
plt.hist(np.random.normal(0,np.sqrt(np.var(V[5,:])),189900),bins=21,log=True,rwidth=0.8)
plt.xlim(-0.04,0.04)
plt.xlabel("$v_6$")
plt.ylabel("number")     

plt.figure(figsize=(12,6))
plt.hist(V[5,:],bins=21,log=True,rwidth=0.8)  
plt.xlabel("$v_6$")
plt.ylabel("number")

'''V6=np.sort(np.abs(V[5,:]))
Vborder=V6[int(V6.size*0.95)]
inter=[]
t1=[]
interval=6000
ran=0.0002
for i in range(30):
        inter.append(V[5,7000+i*interval:7000+(i+1)*interval])
        t1.append(t[7000+i*interval:7000+(i+1)*interval])
for i in range(30):
    plt.figure(figsize=(12,6))
    plt.plot(t1[i],inter[i])'''

attempt=[]
v1cutoff=[]
v6part=[]
v1part=[]
n=0
sikiiti=-0.00007
for i in range(q-10200):
    if V[0,i]>sikiiti and V[0,i+1]>sikiiti:
        v6part.append(V[5,i])
    elif V[0,i]>sikiiti and V[0,i+1]<sikiiti:
        attempt.append(v6part)
        v6part=[]
    elif V[0,i]<sikiiti and V[0,i+1]<sikiiti:
        v1part.append(V[0,i])
    elif V[0,i]<sikiiti and V[0,i+1]>sikiiti:
        v1cutoff.append(v1part)
        v1part=[]    



plt.figure(figsize=(10,10))
ampv1=[]
stdv6=[]
for i in range(len(v1cutoff)):
    ampv1.append(np.min(np.array(v1cutoff[i])))
    stdv6.append(np.sqrt(np.var(np.array(attempt[i]))))
ampv1=np.array(ampv1)
stdv6=np.array(stdv6)
ampv1=ampv1*-1
plt.plot(stdv6,ampv1,".")
plt.ylabel("amplitude of $v_1$")
plt.xlabel("standard deviation of $v_6$")


corr=[]
for i  in range(len(v1cutoff)):
    corrcoef=np.corrcoef(ampv1[:i],stdv6[:i])[0,1]
    corr.append(corrcoef)
plt.rcParams["font.size"] = 16
plt.figure(figsize=(12,8))
plt.plot(corr,".")
plt.xlabel("number of data")
plt.ylabel("correlation coefficient")

s=11
s1=10
sep=np.linspace(-1*s1,s-1,s+s1)
corrsep=[]
for i in range(s1):
    corrcoefsp=np.corrcoef(stdv6[s1-i:len(stdv6)],ampv1[0:len(ampv1)-s1+i])[0,1]
    corrsep.append(corrcoefsp)
for i in range(s):
    corrcoefsp=np.corrcoef(stdv6[:len(stdv6)-i],ampv1[i:])[0,1]
    corrsep.append(corrcoefsp)
plt.figure(figsize=(12,6))
plt.plot(sep,corrsep,".")
plt.xticks(np.arange(-1*s1,s,2))
plt.xlabel("seperation")
plt.ylabel("correlation coefficient")
