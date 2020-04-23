# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:09:57 2019

@author: amanumayuta
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import seaborn as sns

def gaussian(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2 / 2)

def prob_model(a, b, x):
    return (1-a)*gaussian(x) + a * gaussian(x-b)


def prob_model_true(x):
    return (1-A)*gaussian(x) + A * gaussian(x-B)

def prob_params(a, b):
    return 1/ (RANGE_A*RANGE_B)
#真の分布におけるパラメータ
A = 0.5
B = 3
#分配関数の計算をするときの刻み幅
dA = 0.01
dB = 0.01
#パラメータのとる範囲
RANGE_A = 1
RANGE_B = 10
#パラメーターの下限値
low_A = 0
low_B = -5


Beta = 1

K = 3

def sampling(n, k):
    i = 0
    sample = []
    while i < n: 
        samp = np.random.rand()#[0, 1]の一様分布からサンプリングを得る
        samp = 9*samp -3
        pro = np.random.rand()*k
        if pro < prob_model_true(samp):
            i += 1
            sample.append(samp)
    return sample


def bunpai_kansuu(sample, beta = 1.0):
    num_A = int(RANGE_A // dA)+1
    num_B = int(RANGE_B // dB)+1
    prob_table = np.zeros((num_A, num_B))
    A_table = np.linspace(low_A, low_A + RANGE_A, num_A)
    B_table = np.linspace(low_B, low_B + RANGE_B, num_B)
    for i, a in enumerate(A_table):
        for j, b in enumerate(B_table):
            p = prob_model(a, b, sample)
            p = np.array(p)
            p = p ** beta
            p = p.prod()
            prob_table[i, j] = p
    return prob_table, A_table, B_table
    
def jigo_prob(num_sample):
    sample = np.array(sampling(num_sample, K))
    z, a, b = bunpai_kansuu(sample)
    if num_sample % 10 == 0:
        prob = z/ (np.sum(z)* dA * dB)
        draw_3D(a, b, prob,num_sample, ["$w_1$", "$w_2$", "probability"])
    return z, a, b

def draw_3D(x, y, z, n, label = ["x", "y", "z"]):
    plt.rcParams["font.size"] = 25
    plt.rcParams["axes.labelpad"]=20
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize = (12, 8))
    ax = Axes3D(fig)
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    ax.plot_wireframe(X, Y, np.transpose(z), cmap='ocean')
    ax.view_init(elev=45, azim=45)
    #plt.savefig(r'C:\Users\amanuma_yuta\Desktop\bayes\pic\num_sample\figureb=0.5_sample'+str(n)+'.png')
    plt.show()

def calculate_energy_vs_num_sample():
    a = 1
    F = []
    num_sample = []
    for i in range(10, 110, 10):
        print(i)
        z, a, b = jigo_prob(i)
        F.append(-1*np.log(np.sum(z)*dA*dB))
        num_sample.append(i)
    plt.figure(figsize = (12, 6))
    plt.plot(num_sample, F)
    plt.xlabel("number of sample")
    plt.ylabel("free energy")
    plt.show()
    return num_sample, F

def draw_sampler(n):
    x = np.linspace(-4, 6, 1000)
    p = prob_model_true(x)
    plt.figure(figsize = (8, 6))
    sample = sampling(n, 5)
    sns.distplot(sample)
    plt.plot(x, p)

if __name__ == "__main__":
    num_sample, F  = calculate_energy_vs_num_sample()
    #draw_sampler(100)
    
    """
    prob = z/ (np.sum(z)* dA * dB)
    draw_3D(a, b, prob, ["a", "b", "probability"])
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (12, 6))
    plt.xlabel("value of x")
    plt.ylabel("number of sample")
    plt.hist(a, bins = 100)
    """