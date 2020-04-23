# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:29:35 2020

@author: ryuch
"""


import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import japanize_matplotlib
import matplotlib.cm as cm

import csv

#真の分布におけるパラメータ
A = 0.5
B = 0.5
beta = 1.0


def gaussian(x):
    a = np.sqrt(2*np.pi)
    return 1/a *np.exp(-x**2 / 2)

def prob_model(a, b, x):
    return (1-a)*gaussian(x) + a * gaussian(x-b)

def prob_model_true(x):
    return (1-A)*gaussian(x) + A * gaussian(x-B)

def sampling(n, k = 3):
    i = 0
    sample = []
    while i < n: 
        samp = np.random.rand()#[0, 1]の一様分布からサンプリングを得る
        samp = 9*samp -3
        pro = np.random.rand()*k
        if pro < prob_model_true(samp):
            i += 1
            sample.append(samp)
    return np.array(sample)

def draw(n):
    x = np.linspace(-4, 5, 1000)
    p = prob_model_true(x)
    plt.figure(figsize = (8, 6))
    sample = sampling(n, 5)
    sns.distplot(sample)
    plt.plot(x, p)

def hamiltonian(sample, para):
    H = -1*np.sum(np.log(prob_model(para[0], para[1], sample)))
    return H
    
def metropolis(sample,n = 1000, std = 0.6):
    w1 = np.random.rand(2)
    w = [w1]
    count = 0
    while count < n:
        w_temp = [np.random.rand(), np.random.normal(w[-1][1], std)]
        delta_H = hamiltonian(sample, w_temp)-hamiltonian(sample, w[-1])
        P = min(1, np.exp(-1*beta*delta_H))
        p_temp = np.random.rand()
        if p_temp <= P:
            w.append(w_temp)
            count += 1
    return w


def draw_2d_hist(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H = ax.hist2d(x,y, bins=80, cmap=cm.jet)
    ax.set_title('1st graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)

def main(N = 500):
    sample = sampling(N, k = 4)
    w = np.array(metropolis(sample, 1000))
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (8, 8))
    plt.scatter(w[:, 0], w[:, 1], color = '#ff7f00')
    plt.title("事後分布からのサンプリング")
    plt.xlim(0, 1)
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")
    return w

    

if __name__ == "__main__":
    
    #draw(10000)
    w = main(6000)
    #sample = sampling(500)
    #draw_2d_hist(sample[:, 0], sample[:, 1])
    """
    prob = z/ (np.sum(z)* dA * dB)
    draw_3D(a, b, prob, ["a", "b", "probability"])
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (12, 6))
    plt.xlabel("value of x")
    plt.ylabel("number of sample")
    plt.hist(a, bins = 100)
    """