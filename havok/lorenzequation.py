# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:39:27 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from rungekutta import RungeKutta
from mpl_toolkits.mplot3d import Axes3D
#ローレンツ方程式の定義
def function(x):
    return np.array((10*x[1]-10*x[0],x[0]*28-x[0]*x[2]-x[1],x[0]*x[1]-2.6667*x[2]),dtype=float)

#ローレンツ方程式からデータの取得
rungekutta=RungeKutta(function)
x,y=rungekutta.calculation([10,10,10],0.001,200000)
plt.plot(x[:,0])
fig1=plt.figure()
ax=Axes3D(fig1)
ax.plot(x[:,0],x[:,1],x[:,2],".",ms=1.3)

#ハンケル行列の生成
H=[]
for i in range(100):
    H.append(x[i:199900+i,0])
H=np.reshape(H,(100,199900))

H1=[]
for i in range(100):
    H1.append(y[i:199900+i,0])
H1=np.reshape(H1,(100,199900))

#特異値分解
U, S, V = np.linalg.svd(H,full_matrices=False) 
v15=V[14,:]
U1, S1, V1 = np.linalg.svd(H1,full_matrices=False)
fig2=plt.figure()
#ローレンツアトラクターの再構成
ax=Axes3D(fig2)
ax.plot(V[0,:],V[1,:],V[2,:],".",ms=1.3)

#行列の整形
v=np.delete(V,np.arange(15,100),0)
v2=np.delete(V1,np.arange(15,100),0)

#V'の算出
V3=[]
for i in range(199898):
     V3.extend(np.transpose((v[:,i+2]-v[:,i])/(2*0.001)))
V3=np.reshape(V3,(15,199898))
V3=np.delete(V3,np.arange(10000,199898),1)

v=np.delete(v,0,1)
v=np.delete(v,np.arange(10000,199899),1)   
pinvV = np.linalg.pinv(v)
A=np.dot(V3,pinvV)


A1=np.delete(A,14,0)
A1=np.delete(A1,14,1)

B=np.delete(A,np.arange(0,14),1)
B=np.delete(B,14,0)

def function1(A1,x):
    return np.dot(A1,x)
#havokによる計算
rungekutta2=RungeKutta(function1)
havok=rungekutta2.calculation1(v[0:14,0],0.001,v15,B,A1)
