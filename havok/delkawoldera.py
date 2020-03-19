# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:45:48 2018

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from rungekutta import RungeKutta1, RungeKutta3

a=[1,1,1,1]
def f(x):
    return np.array([a[0]*x[0]-a[1]*x[0]*x[1],-a[2]*x[1]+a[3]*x[0]*x[1]])



rungekutta=RungeKutta1(f)
r,A=rungekutta.calculation([1.2,1.5],0.001,200000)
plt.rcParams["font.size"] = 30 
plt.figure(figsize = (12, 8))
plt.plot(r[:,0],r[:,1])
plt.xlabel("$x$")
plt.ylabel("$y$")

from mpl_toolkits.mplot3d import Axes3D   
import matplotlib.pyplot as plt 
fig = plt.figure(figsize = (12, 8)) #プロット領域の作成
ax = fig.gca(projection='3d') #プロット中の軸の取得。gca は"Get Current Axes" の略。

x = np.arange(0.3, 2, 0.01) # x点として[-2, 2]まで0.05刻みでサンプル
y = np.arange(0.3, 2, 0.01)  # y点として[-2, 2]まで0.05刻みでサンプル
x, y = np.meshgrid(x, y)  # 上述のサンプリング点(x,y)を使ったメッシュ生成

z = a[2]*x + a[1]*y -a[3]*np.log(x) -a[0]*np.log(y)  #exp(-(x^2+y^2))を計算してzz座標へ格納する。
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hsv', linewidth=0.3) # 曲面のプロット。rstrideとcstrideはステップサイズ，cmapは彩色，linewidthは曲面のメッシュの線の太さ，をそれぞれ表す。

z = a[2]*r[:, 0] + a[1]*r[:, 1] -a[3]*np.log(r[:, 0]) -a[0]*np.log(r[:, 1])  #exp(-(x^2+y^2))を計算してzz座標へ格納する。
ax.plot(r[:, 0], r[:, 1], z, color = 'b', linewidth = 2.0)
ax.view_init(elev=40, azim=15)

plt.show()