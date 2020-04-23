# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです。
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
B = np.array([0.5, 0.5])
C = np.array([-0.5, -0.5])
beta = 1.0

h = 0.0001
alpha = 1

def gaussian(x, co_var = np.array([[1, 0], [0, 1]])):
    a = np.sqrt(2*np.pi)**(len(x))
    return 1/a *np.exp(-np.dot(x, x.T) / 2)

def prob_model(a, b, c, x):
    return (1-a)*gaussian(x-b) + a * gaussian(x-c)

def prob_model_true(x):
    return (1-A)*gaussian(x-B) + A * gaussian(x-C)

def sampling(n, k = 3):
    i = 0
    sample = []
    while i < n: 
        samp = np.random.rand(2)#[0, 1]の一様分布からサンプリングを得る
        samp = 6*samp -3
        pro = np.random.rand()*k
        if pro < prob_model_true(samp):
            i += 1
            sample.append(samp)
    return np.array(sample)

def pre_distribution_a(a, alpha = 1):
    return (a*(1-a))**(alpha-1)

def pre_ditribution_bc(b, c):
    return np.exp(-1*h/2 *(np.linalg.norm(b)**2 + np.linalg.norm(c)**2))

def dhilicre_dist(a, n):
    return np.prod(a**(alpha - 1 + n))
    

def gibs(sample,N = 500):
    num_data = len(sample)
    y = np.random.dirichlet([1, 1], num_data)
    y = (y > 0.5)*1
    count = 0
    para = []
    while count < N + 20:
        count += 1
        n = np.sum(y, axis = 0)
        #print(n)
        B0 = np.sum((y[:, 0])*sample.T, axis = 1)/(n[0]+h)
        B1 = np.sum(y[:, 1]*sample.T, axis = 1)/(n[1]+h)
        s = 1/(h + n)
        a = np.random.dirichlet(alpha + n)
        b1 = np.random.normal(B0, s[0])
        b2 = np.random.normal(B1, s[1])
        L1k = a[0] * np.exp(-1/2*np.sum((sample-b1)**2, axis = 1 ))
        L2k = a[1] * np.exp(-1/2*np.sum((sample-b2)**2, axis = 1 ))
        p1 = L1k/ (L1k + L2k)
        p2 = L2k/ (L1k + L2k)
        p = np.array([p1, p2]).T
        y =  (np.random.dirichlet([1, 1], num_data) < p)*1
        if 20 < count:
            para_temp = [a[1]]
            para_temp.extend(b1)
            para_temp.extend(b2)
            para.append(para_temp)
    return np.array(para)
        
def main(num_data_sample, num_para_sample):
    sample = sampling(num_data_sample)
    w = gibs(sample, num_para_sample)
    return w
  
def draw_sampling():
    test = [10, 80, 160, 320]
    fig = plt.figure(figsize = (24, 20))
    plt.rcParams["font.size"] = 40
    for i, n in enumerate(test):
        w = main(n, 500)
        ax = fig.add_subplot(2, 2, (i + 1))
        ax.scatter(w[:, 1], w[:, 2])
        ax.scatter(w[:, 3], w[:, 4])
        ax.set_xlabel("$w1, w3$")
        ax.set_ylabel("$w2, w4$")
        ax.set_title("サンプル数{}".format(n))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.show()
      
if __name__ == "__main__":
    
    #draw(10000)
    w = main(100, 700)
    #sample = sampling(200)
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