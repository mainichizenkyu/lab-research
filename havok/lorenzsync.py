# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:36:47 2019

@author: amanuma_yuta
"""
import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from mpl_toolkits.mplot3d import Axes3D
#ローレンツ方程式の定義
def f(x,d):
    return np.array((10*x[1]-10*x[0]+d*(x[3]-x[0]),x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-2.6667*x[2],
                     10*x[4]-10*x[3]+d*(x[0]-x[3]),x[3]*28-x[3]*x[5]-x[4],x[3]*x[4]-2.6667*x[5]),dtype=float)

def calculate(x,d,h):
    s1=f(x,d)
    s2=f(x+0.5*s1*h,d)
    s3=f(x+0.5*s2*h,d)
    s4=f(x+h*s3,d)
    return x+h/6*(s1+2*s2+2*s3+s4),s4
def calculation(x0,h,n,d):
    x=[x0]
    y=[]
    for i in range(n):
        x1,y1=calculate(x0,d,h)
        x.append(x1)
        y.append(y1)
        x0 =x1
    return np.array(x),np.array(y)
    
if __name__=='__main__':
    h=0.001
    x,y=calculation([10,10,10,5,6,6],h,500000,3.7)
    plt.figure(figsize=(12,6))
    plt.plot(x[:,3])
    plt.plot(x[:,0])
    plt.figure(figsize=(6,6))
    plt.plot(x[:,0],x[:,3])
    plt.figure(figsize=(12,6))
    plt.plot(x[:,0]-x[:,3])
    r=12
    cal=Havok(x[:,0]-x[:,3])
    havok1,A1,B1,v1,dv1,vdf,V,S,U=cal.calculation2(100,r,r,h)
    plt.rcParams["font.size"] = 20
    t=np.linspace(0,189.9,189900)
    plt.figure(figsize=(12,6))
    plt.plot(t[150000:175000],V[0,150000:175000],label="v1")
    plt.plot(t[150000:175000],V[11,150000:175000],label="v12")
    plt.xlabel("time(s)")
    plt.ylabel("$v_1$")
    plt.legend()
