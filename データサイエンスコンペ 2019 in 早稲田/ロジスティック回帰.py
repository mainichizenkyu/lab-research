# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:35:00 2019

@author: amanuma_yuta
"""


import pandas as pd


from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#CSVファイルの読み込み
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import lightgbm as lgb
data_2013 = pd.read_csv("data2013_new.csv")
data_2016 = pd.read_csv("data2016_new.csv")

rice2013 = pd.read_csv("rice2013.csv")
rice2016 = pd.read_csv("rice2016.csv")

data_new2013 = data_2013.merge(rice2013, on="選挙区")
data_new2016 = data_2016.merge(rice2016, on="選挙区")
#説明変数以外を消去する, ここでは説明変数は　年齢, 当選回数, 東京からの距離, 政党_世論　の4つを採用
#ターゲット変数は当落の2値
#データの成型


var_2013 = data_new2013[['年齢', '当選回数','地方議席数_割合']]
tar_2013 = data_new2013[['当選']]
var_2016 = data_new2016[['年齢', '当選回数', '地方議席数_割合']]
tar_2016 = data_new2016[['当選']]

        

#2013年のデータの成形
'''
var_2013['年齢'] = (var_2013['年齢']-var_2013['年齢'].mean())/var_2013['年齢'].std()**0.5
var_2013['当選回数'] = (var_2013['当選回数']-var_2013['当選回数'].mean())/var_2013['当選回数'].std()**0.5
var_2013['政党_世論'] = (var_2013['政党_世論']-var_2013['政党_世論'].mean())/var_2013['政党_世論'].std()**0.5
var_2013['地方議席数_割合'] = (var_2013['地方議席数_割合']-var_2013['地方議席数_割合'].mean())/var_2013['年齢'].std()**0.5
var_2013['東京からの距離'] = (var_2013['東京からの距離']-var_2013['東京からの距離'].mean())/var_2013['東京からの距離'].std()**0.5
var_2013['米収穫高t'] = (var_2013['米収穫高t']-var_2013['米収穫高t'].mean())/var_2013['米収穫高t'].std()**0.5
'''
'''
#2016年のデータの成形
var_2016['年齢'] = (var_2016['年齢']-var_2016['年齢'].mean())/var_2016['年齢'].std()**0.5
var_2016['当選回数'] = (var_2016['当選回数']-var_2016['当選回数'].mean())/var_2016['当選回数'].std()**0.5
var_2016['地方議席数_割合'] = (var_2016['地方議席数_割合']-var_2016['地方議席数_割合'].mean())/var_2016['年齢'].std()**0.5
var_2016['政党_世論'] = (var_2016['政党_世論']-var_2016['政党_世論'].mean())/var_2016['政党_世論'].std()**0.5

var_2016['東京からの距離'] = (var_2016['東京からの距離']-var_2016['東京からの距離'].mean())/var_2016['東京からの距離'].std()**0.5
var_2016['米収穫高t'] = (var_2016['米収穫高t']-var_2016['米収穫高t'].mean())/var_2016['米収穫高t'].std()**0.5
'''

#ロジステック回帰モデルの生成
clf = LogisticRegression()
X = var_2013
y = tar_2013
clf.fit(X, y)

clf.score(X,y)

#生成したモデルを用いて別の選挙の予測
predict_2016_log = clf.predict(var_2016)
 
#実際の値と予測値の比率
accuracy_score(tar_2016, predict_2016_log)

#点数の出力
testpre =np.array(tar_2016["当選"])
point_logi = 0
for i in range(len(testpre)):
    if testpre[i] == 1 and predict_2016_log[i]==1:
        point_logi += 1
print('ロジスティック回帰で得られる点数は　%.2f' % point_logi)

#選挙区の当選人数を鑑みて予測する.
def cal_proba_logi(data):
    prediction = pd.DataFrame(index = [], columns = ['政党', '当選'])
    for name, group in data:
        a = int(np.average(group['定員']))
        var = group[['年齢', '当選回数', '地方議席数_割合']]
        probability = clf.predict_proba(var)
        pre_index=np.argpartition(-probability[:,1], a)[:a]
        ans = np.zeros_like(np.argpartition(-probability[:,1], a))
        ans[pre_index] = 1
        pre = group[['当選','政党']]
        pre['予想'] = ans
        prediction=prediction.append(pre)
    return prediction
        
    
index_logi = data_new2016.set_index('選挙区')
index_logi_groupby2 = index_logi.groupby(level=0) 
prediction_logi =cal_proba_logi(index_logi_groupby2)
point_logi = 0
predict_logi=np.array(prediction_logi['予想'])
answer_logi = np.array(prediction_logi['当選'])
for i in range(len(testpre)):
    if predict_logi[i]== 1 and answer_logi[i]==1:
        point_logi += 1
print('ロジスティック回帰で得られる点数(人数調整済)は　%.2f' % point_logi)




#SVMによる予測モデル

modelSVM = SVC (kernel='linear', random_state = None, probability = True)
modelSVM.fit(var_2013, tar_2013)

predict_2013_SVM = modelSVM.predict(var_2013)
accuracy_train_2013 = accuracy_score(tar_2013, predict_2013_SVM)
print('SVMによるトレーニングの正答率 : %.2f' % accuracy_train_2013)

predict_2016_SVM = modelSVM.predict(var_2016)
accuracy_train_2016 = accuracy_score(tar_2016, predict_2016_SVM)
print('SVMによるテストデータの正答率 : %.2f' % accuracy_train_2016)

point_SVM = 0
for i in range(len(testpre)):
    if testpre[i] == 1 and predict_2016_SVM[i]==1:
        point_SVM += 1
print('SVMで得られる点数は　%.2f' % point_SVM)

#選挙区の当選人数を鑑みて予測する.
def cal_proba_SVM(data):
    prediction = pd.DataFrame(index = [], columns = ['政党', '当選'])
    for name, group in data:
        a = int(np.average(group['定員']))
        var = group[['年齢', '当選回数', '地方議席数_割合']]
        probability = modelSVM.predict_proba(var)
        pre_index=np.argpartition(-probability[:,1], a)[:a]
        ans = np.zeros_like(np.argpartition(-probability[:,1], a))
        ans[pre_index] = 1
        pre = group[['当選', '政党']]
        pre['予想'] = ans
        prediction=prediction.append(pre)
    return prediction
        
    
index_logi = data_new2016.set_index('選挙区')
index_logi_groupby2 = index_logi.groupby(level=0) 
prediction_SVM =cal_proba_SVM(index_logi_groupby2)
point_SVM = 0
predict_SVM=np.array(prediction_SVM['予想'])
answer_SVM = np.array(prediction_SVM['当選'])
for i in range(len(testpre)):
    if predict_SVM[i]== 1 and answer_SVM[i]==1:
        point_SVM += 1
print('SVMで得られる点数(人数調整済)は　%.2f' % point_SVM)





#SVMとロジスティック回帰の結果の比較
print(np.sum(predict_2016_log))
print(np.sum(predict_2016_SVM))
point_SVM_logi = 0
for i in range(len(testpre)):
    if predict_logi[i] == 1 and predict_SVM[i]==1:
        point_SVM_logi += 1
print('ロジスティック回帰とSVMの予測の一致率は　%.2f' % point_SVM_logi)





linear_regression = LinearRegression()
linear_regression.fit(var_2013, tar_2013)
accuracy = linear_regression.score(var_2016, tar_2016)
print('重回帰分析による正答率　: %.2f' % accuracy)

predict_jukaiki = linear_regression.predict(var_2016)
point_jukaiki = 0
for i in range(len(testpre)):
    if testpre[i] == 1 and predict_jukaiki[i] == 1:
        point_jukaiki += 1
print('重回帰による点数は　%.2f' % point_jukaiki)




#lightGBMによる予測

lgb_train = lgb.Dataset(var_2013, tar_2013)
lgb_eval = lgb.Dataset(var_2016, tar_2016)

lgbm_params = {'objective':'multiclass' , 'num_class':2}

gbm = lgb.train(lgbm_params, lgb_train, num_boost_round = 50, valid_sets = lgb_eval, early_stopping_rounds = 10)

GBM_pred = gbm.predict(var_2016, num_iteration=gbm.best_iteration)
point_GBM = 0
for i in range(len(testpre)):
    if testpre[i] == 1 and np.argmax(GBM_pred[i]) == 1:
        point_GBM += 1
print('lightGBMによる点数は　%.2f' % point_GBM)



print('ロジスティック回帰で得られる点数(人数調整済)は　%.2f' % point_logi)
print('SVMで得られる点数(人数調整済)は　%.2f' % point_SVM)
print('ロジスティック回帰とSVMの予測の一致率は　%.2f' % point_SVM_logi)
