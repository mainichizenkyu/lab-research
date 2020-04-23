# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:53:25 2020

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
A = 0.5 #w0
w1 = 0.5
w2 = 0.5
w3 = -0.5
w4 = -0.5

B = np.array([w1, w2])
C = np.array([w3, w4])
T_para =  [A, w1, w2, w3, w4]

beta = 1.0#逆温度パラメータ

num_pos = 500

def gaussian_pos(x, co_var = np.array([[1, 0], [0, 1]])):
    a = np.sqrt(2*np.pi)**(len(x[0, :]))
    return 1/a *np.exp(- np.sum(x*x, axis = 1) / 2)

def gaussian_sample(x, co_var = np.array([[1, 0], [0, 1]])):
    a = np.sqrt(2*np.pi)**(len(x))
    return 1/a *np.exp(- np.dot(x, x.T)/ 2)

def prob_model(a, b, c, x):
    return (1-a)*gaussian_pos(x-b) + a * gaussian_pos(x-c)

def prob_model_true(x):
    return (1-A)*gaussian_sample(x-B) + A * gaussian_sample(x-C)

def draw_prob_true(num_sample=200, num_cut = 200):
    X = np.linspace(-3, 3, num_cut)
    Y = np.linspace(-3, 3, num_cut)
    P = np.zeros((num_cut, num_cut))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            P[j, i] = prob_model_true(np.array([x, y])) 
    X, Y = np.meshgrid(X, Y)
    plt.rcParams["font.size"] = 25
    plt.rcParams["axes.labelpad"]=20
    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, P, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("probability density")
    ax.view_init(elev=50, azim=80)


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

def draw(n):
    x = np.linspace(-4, 5, 1000)
    p = prob_model_true(x)
    plt.figure(figsize = (8, 6))
    sample = sampling(n, 5)
    sns.distplot(sample)
    plt.plot(x, p)

def hamiltonian(sample, para):
    H = -1*np.sum(np.log(prob_model(para[0], np.array(para[1:3]),np.array(para[3:5]), sample)))
    return H
    
def metropolis(sample,n = 1000, std = 0.05, burn_in = 200):
    w1 = [np.random.rand()]
    w1.extend(np.random.rand(4))
    w = [w1]
    count = 0
    cov = [[std, 0, 0, 0], [0, std, 0, 0], [0, 0, std, 0], [0, 0, 0, std]]
    while count < n + burn_in:
        w_temp = [np.random.rand()]
        w_temp.extend(np.random.multivariate_normal(w[-1][1:], cov))
        delta_H = hamiltonian(sample, w_temp)-hamiltonian(sample, w[-1])
        P = min(1, np.exp(-1*beta*delta_H))
        p_temp = np.random.rand()
        if p_temp <= P:
            #print("Sample!!")
            #print(w_temp)
            w.append(w_temp)
            count += 1
    return np.array(w[burn_in:])


def draw_2d_hist(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H = ax.hist2d(x,y, bins=80, cmap=cm.jet,  normed = True)
    ax.set_title('distribution')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.colorbar(H[3],ax=ax)


def draw_hamiltonian(num_sample=200, num_cut = 200):
    sample = sampling(70, k = 4)
    W3 = np.linspace(-3, 3, num_cut)
    W4 = np.linspace(-3, 3, num_cut)
    H = np.zeros((num_cut, num_cut))
    for i, w3 in enumerate(W3):
        for j, w4 in enumerate(W4):
            H[j, i] = hamiltonian(sample, [0.5, w3, w4, -0.5 , -0.5]) 
    w3, w4 = np.meshgrid(W3, W4)
    plt.rcParams["font.size"] = 25
    plt.rcParams["axes.labelpad"]=20
    fig = plt.figure(figsize = (24, 8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot_surface(w3, w4, H, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax1.set_xlabel("$w_1$")
    ax1.set_ylabel("$w_2$")
    ax1.set_zlabel("hamiltonian")
    ax1.view_init(elev=60, azim=80)
    
    H = np.zeros((num_cut, num_cut))
    for i, w3 in enumerate(W3):
        for j, w4 in enumerate(W4):
            H[j, i] = hamiltonian(sample, [0.5, w3, 0.5, w4 , -0.5]) 
    w3, w4 = np.meshgrid(W3, W4)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_surface(w3, w4, H, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax2.set_xlabel("$w_1$")
    ax2.set_ylabel("$w_3$")
    ax2.set_zlabel("hamiltonian")
    ax2.view_init(elev=60, azim=80)
    


def emp_log_loss_func(sample, para):
    return -1/len(sample)*np.sum(np.log(prob_model(para[0], np.array(para[1:3]),np.array(para[3:5]), sample)))

def sample_pos(N = 500):
    sample = sampling(N, k = 4)
    w = np.array(metropolis(sample, 2000))
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (8, 8))
    plt.scatter(w[:, 1], w[:, 2])
    plt.scatter(w[:, 3], w[:, 4])
    plt.title("事後分布からのサンプリング")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("$w_1$")
    plt.ylabel("$w_2$")
    return w

def AIC(test, n = 10):
    AIC=[]
    for i in test:
        AIC_temp = np.zeros(n)
        for j in range(n):
            sample = sampling(i)
            AIC_temp[j] =  emp_log_loss_func(sample, T_para) + 5/i
        AIC.append([np.mean(AIC_temp), np.std(AIC_temp)])
    return AIC

def prob_model_simple(para, x):
    return (1-para[:, 0])*gaussian_pos(x-para[:, 1:3]) + para[:,0] * gaussian_pos(x-para[:, 3:5])

def GE_test(num_sample = 100):#汎化誤差の計算におけるモンテカルロ法の収束を調べる
    sample_for_pos = sampling(num_sample)
    N = list(range(50, 500, 50))#データの生成確率側
    K = list(range(50, 500, 50))#事後分布からのサンプル
    sample_monte = sampling(500)
    GE = np.zeros((10, 10))
    for i, n in enumerate(N):
        for j, k in enumerate(K):
            E = np.zeros(n)
            w = metropolis(sample_for_pos, k)
            for count, l in enumerate(sample_monte[:n]):
                Ew = 1/k*np.sum(prob_model_simple(w, l))
                E[count] = np.log(Ew)
            GE[i, j] = np.sum(E)/n
    return GE

def GE_calc(num_sample = 100):
    num_q = 250
    print(num_sample)
    sample_for_pos = sampling(num_sample)
    sample_monte = sampling(num_q)
    w = metropolis(sample_for_pos, num_pos)
    E = np.zeros(num_q)
    for i, xi in enumerate(sample_monte):
        E[i] = np.log(1/num_pos*np.sum(prob_model_simple(w, xi)))
    return -1*np.sum(E)/num_q
    
def GE(test, n = 10):
    GE = []
    for i in test:
        ge = np.zeros(n)
        for j in range(n):
            ge[j] = GE_calc(i)
        GE.append([np.mean(ge), np.std(ge)])
    return GE

def Tn_calc(num_sample = 100, num_pos = 250):
    sample_for_pos = sampling(num_sample)
    w = metropolis(sample_for_pos, num_pos)
    E = np.zeros(num_sample)
    for i, xi in enumerate(sample_for_pos):
        E[i] = np.log(1/num_pos*np.sum(prob_model_simple(w, xi)))
    return -1*np.sum(E)/num_sample

def WAIC(test,n = 10):
    WAIC = []
    Tn = []
    Vn = []
    for i in test:
        tn = np.zeros(n)
        vn = np.zeros(n)
        waic = np.zeros(n)
        for j in range(n):
            tn_temp = Tn_calc(i)
            vn_temp = Vn_calc(i)
            waic[j] = tn_temp+beta*vn_temp/i
            tn[j] = tn_temp
            vn[j] = vn_temp
        WAIC.append([np.mean(waic), np.std(waic)])
        Tn.append([np.mean(tn),np.std(tn)])
        Vn.append([np.mean(vn), np.std(vn)])
    return WAIC, Tn, Vn

def log_likely_func(sample, w):
    return np.log(prob_model_true(sample)/prob_model_simple(w, sample))


def Vn_calc(num_sample = 100, num_pos = 250):
    sample_for_pos = sampling(num_sample)
    w = np.array(metropolis(sample_for_pos, num_pos))
    V = np.zeros(num_sample)
    for i, xi in enumerate(sample_for_pos):
        V[i] = np.sum(log_likely_func(xi, w)**2)/num_pos - np.sum(log_likely_func(xi, w)/num_pos)**2
    return np.sum(V)/num_sample

def draw_para():
    test = [5, 10, 20, 40, 80, 160, 240, 320, 640, 1280]
    waic, tn, vn = np.array(WAIC(test, 15))
    aic = np.array(AIC(test, 15))
    ge = np.array(GE(test, 15))
    
    plt.rcParams["font.size"] = 35
    plt.figure(figsize = (12, 8))
    
    plt.errorbar(test, waic[:, 0], yerr = waic[:, 1],capsize = 5, marker = 'o', label = "WAIC")
    plt.errorbar(test, tn[:, 0], yerr = tn[:, 1],capsize = 5, marker = 'v',label = "Tn")
    plt.errorbar(test, ge[:, 0], yerr = ge[:, 1],capsize = 5, marker = 's', label = "GE")
    plt.errorbar(test, aic[:, 0], yerr = ge[:, 1],capsize = 5, marker = '*', label = "AIC")
    
    plt.xlabel("データ数 $n$")
    plt.ylabel("諸値 (無次元)")
    plt.xscale('log')
    plt.grid(linestyle="dashed")
    plt.legend()
    
    return waic, tn ,vn, aic, ge

def draw_sampling():
    test = [5, 10, 20, 40]
    fig = plt.figure(figsize = (24, 20))
    plt.rcParams["font.size"] = 40
    for i, n in enumerate(test):
        w = metropolis(sampling(n))
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
    
    
    #waic, tn, vn, aic, ge = draw_para()
    
    #draw(10000)
    #w = main(300)
    
    #AIC = AIC()
    #sample = sampling(10000)
    #draw_2d_hist(sample[:, 0], sample[:, 1])
    
    #draw_hamiltonian(200, 20)
    
    """
    prob = z/ (np.sum(z)* dA * dB)
    draw_3D(a, b, prob, ["a", "b", "probability"])
    plt.rcParams["font.size"] = 30
    plt.figure(figsize = (12, 6))
    plt.xlabel("value of x")
    plt.ylabel("number of sample")
    plt.hist(a, bins = 100)
    """