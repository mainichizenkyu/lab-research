# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:44:46 2019

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from havok import Havok
from rungekutta import RungeKutta1,RungeKutta3
import scipy as sp
import scipy.fftpack
import random
from scipy.special import perm, comb

N=200000
h=0.001
b=8/3
#ローレンツ方程式の定義 :
def functionlorenz(x):
    return np.array((10*x[1]-10*x[0],x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-2.66666667*x[2]),dtype=float)
rungekutta=RungeKutta1(functionlorenz)
#ランダムな初期値に対してローレンツ方程式の計算:
def calculation(h):
    x,y=rungekutta.calculation([random.uniform(-20,20),random.uniform(-20,20),random.uniform(0,40)],0.001,200000)
    return x

#解析的な公式を用いての高階微分の算出:
def higherderivative(n):
    x=calculation(h)
    dx=[x[:,0]]
    dy=[x[:,1]]
    dz=[x[:,2]]
    for i in range(n):
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
    return x, dx
#遷移点及び2つ手前の極値を取るタイミングの推定:
def check(x):
    jump=[]
    cut=[]
    for i  in range(N-20000):
        if x[i]>0 and x[i+1]<0:
            jump.append(i)
            s=1
            while True:
                s += 1
                if x[i-s-1]>x[i-s] and x[i-s+1]>x[i-s]:
                    cut.append(i-s)
                    break
                elif x[i-s-1]<0 and x[i-s]>0:
                    cut.append(i-s)
                    break
                elif i-s-1<0:
                    cut.append(0)
                    break
        elif x[i]<0 and x[i+1]>0:
            jump.append(i)
            s=1
            while True:
                s += 1
                if x[i-s-1]<x[i-s] and x[i-s+1]<x[i-s]:
                    cut.append(i-s)
                    break
                elif x[i-s-1]>0 and x[i-s]<0:
                    cut.append(i-s)
                    break
                elif i-s-1<0:
                    cut.append(0)
                    break
    return jump, cut
#遷移点と手前の極値の間で高階部分の振幅が閾値を超えているかの判定。そこから突発現象の予測の精度の算出を実施する。:
def prediction(jump,cut,dx):
    fail=0
    success=0
    for i in range(len(jump)):
        if np.max(np.abs(dx[cut[i]:jump[i]]))>3*10**(13):
            success+=1
        else:
            fail+=1
    return success, fail
        
    

if __name__ =='__main__':
   #パラメーターの定義:
   
   n=9
   iteration=50
   success=0
   fail=0
   for j in range(iteration):
       x,dx=higherderivative(n)
       x=x[20000:,0]
       dx=dx[n]
       dx=dx[20000:]
       jump,cut=check(x)
       a, b=prediction(jump,cut,dx)
       success+=a
       fail+=b
   p=success/(success+fail)*100
   print(p)
       
       