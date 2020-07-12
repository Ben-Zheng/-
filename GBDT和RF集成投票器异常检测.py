# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:34:11 2020

@author: 86178
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold,cross_val_score

def datetime_exp(data):
    date_set=[pd.datetime.strptime(dates,'%Y-%m-%d') for dates in data['order_date']]
    data['weekday_data']=[data.weekday() for data in date_set]
    data['daysinmonth_data']=[data.day for data in date_set]
    data['month_data']=[data.month for data in date_set]
    
    time_set=[pd.datetime.strptime(times,'%H:%M:%S') for times in data['order_time']]
    data['second_data']=[data.second for data in time_set]
    data['minute_data']=[data.minute for data in time_set]
    data['hour_data']=[data.hour for data in time_set]
    return data.drop(['order_date','order_time'],axis=1)

raw_data=pd.read_table('abnormal_orders.txt',delimiter=',')

raw_data.describe().round(2).T

#缺失值检查
na_cols=raw_data.isnull().any(axis=0)
print('{:*^60}'.format('NA cols:'))
print(na_cols[na_cols==True])
#类样本均衡审查
print(raw_data.iloc[:,-1].value_counts())
#数据预处理
drop_na_set=raw_data.dropna()
drop_na_set=drop_na_set.drop(['order_id'],axis=1)
#字符串转数值
convert_cols=['cat','attribution','pro_id','pro_brand','order_source','pay_type','user_id','city']
enc=OrdinalEncoder()
drop_na_set[convert_cols]=enc.fit_transform(drop_na_set[convert_cols])
#分开测试集与训练集
data_final = datetime_exp(drop_na_set)
num=int(0.7*data_final.shape[0])
x_raw,y_raw=data_final.drop(['abnormal_label'],axis=1),data_final['abnormal_label']
x_train,x_test=x_raw.iloc[:num,:],x_raw.iloc[num:,:]
y_train,y_test=y_raw.iloc[:num],y_raw.iloc[num:]
#样本均衡
model_smote=SMOTE()
x_smote_resampled,y_smote_resampled=model_smote.fit_sample(x_train,y_train)
#模型训练
model_rf=RandomForestClassifier(max_features=0.8,random_state=0)#随机森林
model_gdbc=GradientBoostingClassifier(max_features=0.8,random_state=0)#梯度提升树
estimators=[('randomforest',model_rf),('gradientboosting',model_gdbc)]

model_vot=VotingClassifier(estimators=estimators,voting='soft',weights=[0.9,1.2],n_jobs=-1)#投票器
cv=StratifiedKFold(5,random_state=2)#交叉验证
cv_score=cross_val_score(model_gdbc,x_smote_resampled,y_smote_resampled,cv=cv)

print('{:*^60}'.format('Cross val scores:'),'\n',cv_score)
print('Mean scores is :%.2f'%cv_score.mean())

model_vot.fit(x_smote_resampled,y_smote_resampled)
x_new=pd.read_csv('new_abnormal_orders.csv')
x_new_drop=x_new.drop(['order_id'],axis=1)
x_new_drop[convert_cols]=enc.transform(x_new_drop[convert_cols])
x_new_final=datetime_exp(x_new_drop)
#预测结果
predict_label=model_vot.predict(x_new_final)
predict_proba=model_vot.predict_proba(x_new_final)
predict_np=np.hstack((predict_label.reshape(-1,1),predict_proba))
predict_pd=pd.DataFrame(predict_np,columns=['label','proba_0','proba_1'])
print('{:*^60}'.format('predicted labels:'),'\n',predict_pd)











































































