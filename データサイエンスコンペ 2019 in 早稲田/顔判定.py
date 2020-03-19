# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:40:25 2019

@author: amanuma_yuta
"""

import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from PIL import Image
import os
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


#pic2013内にある画像のファイル名を取得
files_2013 = glob.glob("pic2013/*")
#ファイル名のうちディレクトリのパスを消去
for i in range(len(files_2013)):
    files_2013[i] = files_2013[i].lstrip("pic2013")
    files_2013[i] = files_2013[i].lstrip('\\')
    files_2013[i] = files_2013[i].rstrip(".jpeg")

#pic2016内にある画像のファイル名を取得
files_2016 = glob.glob("pic2016/*")
#ファイル名のうちディレクトリのパスを消去
for i in range(len(files_2016)):
    files_2016[i] = files_2016[i].lstrip("pic2016")
    files_2016[i] = files_2016[i].lstrip('\\')
    files_2016[i] = files_2016[i].rstrip(".jpeg")

images2013 = os.listdir("pic2013")

batch_list_train = []

for i in images2013:
    if i == "Thumbs.db":
        continue
    
    image = np.array(Image.open("pic2013/"+i))
    b = image.tolist()
    batch_list_train.append(b)

images2016 = os.listdir("pic2016")

batch_list_test = []

for i in images2016:
    if i == "Thumbs.db":
        continue
    
    image = np.array(Image.open("pic2016/"+i))
    b = image.tolist()
    batch_list_test.append(b)


'''
for f in files:
  img = cv2.imread(f, cv2.IMREAD_COLOR)
''' 
train = pd.read_csv("a.csv")
test = pd.read_csv("b.csv")
  #説明変数以外を消去する, ここでは説明変数は　年齢, 当選回数, 東京からの距離(ここは人口や人口密度なんてどうでしょう), 政党_世論　の4つを採用
train_var = train.drop(['名', '姓', '年度', '当選','推薦・支持', '政党','新旧' ,'略歴', 
                        '選挙区', '選挙方法', '得票数','得票数(割合)', '定員'], axis = 1)
train_pre = train.drop(['名', '姓', '年度','年齢','当選回数', '当選','推薦・支持', '政党','新旧' ,'略歴', 
                        '選挙区', '選挙方法', '得票数','東京からの距離', '政党_世論', '定員'], axis = 1)
test = test.replace(np.nan, '', regex = True)
    
#教師ラベルの取得
train['姓名'] = train['姓']+train['名'] 
test['姓名'] = test['姓']+test['名']
label2013 = []
for i in range(len(files_2013)):
    a = train[train['姓名']== files_2013[i]]['当選']
    if len(a) == 0:
        pass
    else:
        label1 = int(a)     
    label2013.append(label1)
    
label2016 = []
for i in range(len(files_2016)):
    a = test[test['姓名']== files_2016[i]]['当選']
    if len(a) == 0:
        pass
    else:
        label2 = int(a) 
    label2016.append(label2)
   
batch_list_train = np.array(batch_list_train)
batch_list_test = np.array(batch_list_test)
   
num_class = 2

#データの成型
label_train = keras.utils.to_categorical(label2013, num_class)
pic_train = batch_list_train
label_test = keras.utils.to_categorical(label2016, num_class)
pic_test = batch_list_test

batch_list_train = batch_list_train.astype('float32')
batch_list_test = batch_list_test.astype('float32')

batch_list_train /= 255
batch_list_test /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = (45, 45, 3)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

model.fit(pic_train, label_train, epochs = 2, verbose = 1, validation_data = (pic_test, label_test ) )
score = model.evaluate(pic_test, label_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
