# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:24:01 2019

@author: amanuma_yuta
"""

import numpy as np
import matplotlib.pyplot as plt

def transformation(x, k):
    y1 = (x[0]+ k*np.sin(x[0]+ x[1]))%(2*np.pi)
    y2 = (x[0]+x[1])%(2*np.pi)
    y = [y1-np.pi, y2-np.pi]
    return y

#初期値
cut = 100
xfinal = np.pi
xstart = -1*np.pi
xbegin = np.linspace(xstart, xfinal, cut)
x_total = []
for i in range(cut):
    x0 = [xbegin[i], 0]
    x = [x0]
    k = 0.3
    n = 2000
    for j in range(n):
        x_next = transformation(x0, k)
        x.append(x_next)
        x0 = x_next
    x_total.append(x)
x_total = np.array(x_total)
plt.figure(figsize = (6, 6))
for i in range(cut):
    plt.plot(x_total[i, :, 0], x_total[i, :, 1], ".", ms = 0.3)
plt.xlim(-1*np.pi, np.pi)
plt.ylim(-1*np.pi, np.pi)