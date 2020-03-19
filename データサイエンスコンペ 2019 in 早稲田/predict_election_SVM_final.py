# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:13:28 2019

@author: amanuma_yuta
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.svm import SVC

data_2016 = pd.read_csv("data2016_new.csv")
data_2019 = pd.read_csv("data2019_new.csv")

rice2016 = pd.read_csv("rice2016.csv")
rice2019 = pd.read_csv("rice2019.csv")

data_new2016 = data_2016.merge(rice2016, on="選挙区")
data_new2019 = data_2019.merge(rice2019, on="選挙区")

#教師ラベルとして当落の2値を使用
tar_2016 = data_new2016[['当選']]


#複数人の選挙区の予測における説明変数の選択
var_2016_multiple = data_new2016[['年齢', '当選回数','地方議席数_割合']]
var_2019_multiple = data_new2019[['年齢', '当選回数', '地方議席数_割合']]

#SVMによる予測モデル(定員2人以上)
modelSVM_multiple = SVC(kernel='linear', random_state = None, probability = True)
modelSVM_multiple.fit(var_2016_multiple, tar_2016)


def cal_proba_SVM_multi(data):
    prediction = pd.DataFrame(index = [], columns = ['政党','姓','名'])
    for name, group in data:
        a = int(np.average(group['定員']))
        if a == 1:
            pass
        else:
            var = group[['年齢', '当選回数', '地方議席数_割合']]
            probability = modelSVM_multiple.predict_proba(var)
            pre_index=np.argpartition(-probability[:,1], a)[:a]
            ans = np.zeros_like(np.argpartition(-probability[:,1], a))
            ans[pre_index] = 1
            pre = group[['政党','姓','名']]
            pre['予想'] = ans
            prediction=prediction.append(pre)
    return prediction

index_logi = data_new2019.set_index('選挙区')
index_logi_groupby2 = index_logi.groupby(level=0) 
prediction_SVM_multiple =cal_proba_SVM_multi(index_logi_groupby2)

#1人区の選挙区の予測における説明変数の選択
var_2016_solo = data_new2016[['年齢', '当選回数','地方議席数_割合','米収穫高t']]
var_2019_solo = data_new2019[['年齢', '当選回数','地方議席数_割合','米収穫高t']]


for val in ["当選回数", "年齢", "地方議席数_割合", "米収穫高t"]:
    var_2016_solo[val] = (var_2016_solo[val]-var_2016_solo[val].mean())/var_2016_solo[val].std()
    var_2019_solo[val] = (var_2019_solo[val]-var_2019_solo[val].mean())/var_2019_solo[val].std()


modelSVM_solo = SVC(kernel='linear', random_state = None, probability = True)
modelSVM_solo.fit(var_2016_solo, tar_2016)


def cal_proba_SVM_solo(data):
    prediction = pd.DataFrame(index = [], columns = ['政党','姓', '名'])
    for name, group in data:
        a = int(np.average(group['定員']))
        if a == 1:
            var = group[['年齢', '当選回数', '地方議席数_割合','米収穫高t']]
            probability = modelSVM_solo.predict_proba(var)
            pre_index=np.argpartition(-probability[:,1], a)[:a]
            ans = np.zeros_like(np.argpartition(-probability[:,1], a))
            ans[pre_index] = 1
            pre = group[['政党','姓','名']]
            pre['予想'] = ans
            prediction=prediction.append(pre)
        else:
            pass
    return prediction


index_logi = data_new2019.set_index('選挙区')
index_logi_groupby2 = index_logi.groupby(level=0) 
prediction_SVM_solo =cal_proba_SVM_solo(index_logi_groupby2)

#予測結果の結合
prediction_SVM = pd.concat([prediction_SVM_multiple, prediction_SVM_solo])

#データのまとめ
data_predict = pd.DataFrame({
    "candidate_J": prediction_SVM["姓"]+prediction_SVM["名"],
    "outcome": prediction_SVM["予想"]
})
    
candidates = pd.read_csv("candidates.csv")
candidates = candidates[["num","district_J","district_E","candidate_J","candidate_yomi","candidate_E"]]
candidates_new = candidates.merge(data_predict, on="candidate_J")

candidates_new.to_csv("candidates_submit.csv", index=None)