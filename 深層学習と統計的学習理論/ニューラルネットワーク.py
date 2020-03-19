# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:16:03 2019

@author: amanuma_yuta
"""

PATH = r"C:\Users\amanuma_yuta\.spyder-py3\mine\物性と深層学習\train.csv"
import pandas as pd
import matplotlib.pyplot as plt
#df = pd.read_csv(PATH)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger

import csv
import numpy as np
from sklearn.model_selection import train_test_split


"----------------パラメーター------------------"
batch_size = 128
epochs = 20

"------------------------------------------"



#データの取得
train_label = []
train_data = []
with open(PATH, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            pass
        else:
            train_label.append(row.pop(0))
            row = list(map(int, row))
            train_data.append(row)
train_data = np.array(train_data)
train_label = list(map(int, train_label))
train_label = np.array(train_label)
n_label = len(np.unique(train_label))  # 分類クラスの数 = 5
train_label = np.eye(n_label)[train_label] 
(x_train, x_test, y_train, y_test) = train_test_split(train_data, train_label)


"---------------------Keras ------------------------------------------------------------"

def build_model(input_dim, model_str):
    model = Sequential()
    model.add(Dense(model_str.pop(0), input_dim = input_dim, activation = 'sigmoid'))
    for i, a in enumerate(model_str):
        if not i == len(model_str)-1:
            model.add(Dense(a, activation = 'sigmoid'))
        else:
            model.add(Dense(a, activation = "softmax"))
    model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
    return model

def learning(model_str):
    model = build_model(model_str[0], model_str[1:])
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs , verbose = 1, validation_data = (x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose = 0)
    return score

"---------------------------------手作り---------------------------------------------"





def make_model():#重みの設計
    

def search():
    #層の構造をある範囲内において全探索させる.
    
def plot():
    #層の構造に対して正解率や誤差関数を可視化する.

def phy():
    #得られた重さに対して様々な秩序パラメータを計算する. 

def math():
    #得られた結果に対して代数幾何的な演算を行い, 変換する. 

if __name__=="__main__":
    a = learning([784, 512, 64, 10])
    





