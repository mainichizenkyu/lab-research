# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:09:57 2019

@author: amanumayuta
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def gaussian(x):
    return 1/np.sqrt(np.pi)*np.exp(-x**2 / 2)

def prob_model(a, b, x):
    return (1-a)*gaussian(x) + a * gaussian(x-b)

def prob_params(a, b):
    return 1/ (RANGE_A*RANGE_B)
#真の分布におけるパラメータ
A = 0.5
B = 1.0
#分配関数の計算をするときの刻み幅
dA = 0.005
dB = 0.005
#パラメータのとる範囲
RANGE_A = 1
RANGE_B = 10
#パラメーターの下限値
low_B = -5
low_A = 0

Beta = 1

def sampling(N):
    sample = []
    for i in range(N):
        bound = np.random.rand()
        if bound > A:
            x = np.random.normal(B, 1)
            sample.append(x)
        else:
            x = np.random.randn()
            sample.append(x)
    return sample

def bunpai_kansuu(sample, beta = 1.0):
    num_A = int(RANGE_A // dA)+1
    num_B = int(RANGE_B // dB)+1
    prob_table = np.zeros((num_A, num_B))
    A_table = np.linspace(low_A, low_A + RANGE_A, num_A)
    B_table = np.linspace(low_B, low_B + RANGE_B, num_B)
    for i in range(num_A):
        for j in range(num_B):
            a = low_A + i * dA
            b = low_B + j * dB
            p = prob_model(a, b, sample)
            p = np.array(p)
            p = p ** beta
            p = p.prod()
            prob_table[i, j] = p
    return prob_table, A_table, B_table
    
def jigo_prob(num_sample):
    sample = np.array(sampling(num_sample))
    z, a, b = bunpai_kansuu(sample)
    if num_sample % 50 == 0:
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
    ax.plot_surface(X, Y, np.transpose(z), cmap='ocean')
    ax.view_init(elev=45, azim=45)
    plt.savefig(r'C:\Users\amanuma_yuta\Desktop\bayes\pic\num_sample\figureb=0.5_sample'+str(n)+'.png')
    plt.show()

def calculate_energy_vs_num_sample():
    a = 1
    F = []
    num_sample = []
    with open(r'C:\Users\amanuma_yuta\Desktop\bayes\pic\num_sample\test2.csv','a') as f:
        writer = csv.writer(f)
        for i in range(10, 630, 10):
            print(i)
            z, a, b = jigo_prob(i)
            F.append(-1*np.log(np.sum(z)*dA*dB))
            num_sample.append(i)
            writer.writerow([i,-1*np.log(np.sum(z)*dA*dB) ])
    plt.figure(figsize = (12, 6))
    plt.plot(num_sample, F)
    plt.xlabel("number of sample")
    plt.ylabel("free energy")
    plt.show()
    return num_sample, F

if __name__ == "__main__":
    num_sample, F  = calculate_energy_vs_num_sample()
    
    
    """
    prob = z/ (np.sum(z)* dA * dB)
    draw_3D(a, b, prob, ["a", "b", "probability"])
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (12, 6))
    plt.xlabel("value of x")
    plt.ylabel("number of sample")
    plt.hist(a, bins = 100)
    """