
# coding: utf-8

# #                 CRM Homework (빅데이터 애널리틱스)  by 이택윤

# # Question 1

# In[398]:


import pandas as pd 
import numpy as np 
from pandas import DataFrame , Series
import matplotlib.pyplot as plt
import mglearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression

data =pd.read_csv('C:/Users/renz/Desktop/2힉기/CRM/mailorder.csv', engine='python')   

train = data.iloc[ 0 : 2000, ]
test = data.iloc[ 2000 : ] 
x_train , x_test , y_train, y_test = train_test_split(test[['id','gender','monetary',
                                                            'recency','frequency'  
                                                             ,'duration']], test[['purchase']],  
                                                            random_state =999 , test_size = 0.25)  

print('The number of customers who actually made purchases :' 
                                                      ,pd.concat([x_test,y_test] , axis =1 )['purchase'].sum())
print('ratio :',  pd.concat([x_test,y_test] , axis =1 )['purchase'].sum()/500)


# # Question 2

# In[365]:


def application(train) :        
    if train['recency'] > 12 :   
             r = '1'  
    else  : 
             r = '2'  
    if train['frequency'] < 3 : 
             f = '1'
    else  : 
             f = '2'
    if train['monetary'] < 209 : 
             m = '1'
    else  : 
             m = '2'
    return str(r) + str(f) + str(m)
       
train['RFM'] = train.apply( application, axis = 1 )  

per = train.groupby(['RFM']).sum()['purchase'] / train.groupby(['RFM']).count()['purchase']    
train['percentage'] = train['RFM'].map(per)

def application(test) :      
    if test['recency'] > 12 : 
             r = '1'
    else  : 
             r = '2'
    if test['frequency'] < 3 : 
             f = '1'
    else  : 
             f = '2'
    if test['monetary'] < 209 : 
             m = '1'
    else  : 
             m = '2'   
    return str(r) + str(f) + str(m)
       
test['RFM'] = test.apply( application, axis = 1 )
test['percentage'] = test['RFM'].map(per)

val500 = test.sort_values(by = 'percentage', ascending = False).reset_index().iloc[0:500 ,]
print('The number of customers who actually made purchases :' , val500['purchase'].sum())  
print('ratio :', val500['purchase'].sum()/500)


# # Question 3

# In[366]:


def application(train) :      
    if train['recency'] >16  : 
           r = '1'
    elif 12 < train['recency'] <= 16 : 
           r = '2' 
    elif 8 < train['recency'] <= 12 : 
           r = '3'
    elif 4 < train['recency'] <= 8 : 
           r = '4'
    else : 
           r = '5'
            
    if train['frequency'] == 1  : 
           f = '1'
    elif train['frequency'] == 2 : 
           f = '2' 
    elif 2 < train['frequency'] <= 5 : 
           f = '3'
    elif 5 < train['frequency'] <= 9 : 
           f = '4'
    else : 
           f = '5'
            
    if train['monetary'] <=113   : 
           m = '1'
    elif 113 <= train['monetary'] <= 181 : 
           m = '2' 
    elif 181 < train['monetary'] <= 242 : 
           m = '3'
    elif 242 < train['monetary'] <= 299 : 
           m = '4'
    else : 
           m = '5'
           
    return str(r) + str(f) + str(m)
       
train['RFM'] = train.apply( application, axis = 1 )

per = train.groupby(['RFM']).sum()['purchase'] / train.groupby(['RFM']).count()['purchase']  
train['percentage'] = train['RFM'].map(per)
                                                                                           
def application(test) :      
    if test['recency'] >16  : 
           r = '1'
    elif 12 < test['recency'] <= 16 : 
           r = '2' 
    elif 8 < test['recency'] <= 12 : 
           r = '3'
    elif 4 < test['recency'] <= 8 : 
           r = '4'
    else : 
           r = '5'
            
    if test['frequency'] == 1  : 
           f = '1'
    elif test['frequency'] == 2 : 
           f = '2' 
    elif 2 < test['frequency'] <= 5 : 
           f = '3'
    elif 5 < test['frequency'] <= 9 : 
           f = '4'
    else : 
           f = '5'
            
    if test['monetary'] <=113   : 
           m = '1'
    elif 113 <= test['monetary'] <= 181 : 
           m = '2' 
    elif 181 < test['monetary'] <= 242 : 
           m = '3'
    elif 242 < test['monetary'] <= 299 : 
           m = '4'
    else : 
           m = '5'
           
    return str(r) + str(f) + str(m)
       
test['RFM'] = test.apply( application, axis = 1 )
 
test['percentage'] = test['RFM'].map(per)

val500 = test.sort_values(by = 'percentage', ascending = False).reset_index().iloc[0:500 ,]
print('The number of customers who actually made purchases :' ,val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 5x5x5 RFM codes를 적용했을때 실제 제품을 구매한 결과를 보면 2x2x2 RFM codes를 적용 했을 때보다 15명이 적습니다.
# 
# 결과적으로 5x5x5 RFM codes를 쓴다고해서 성능이 향상되지는 않습니다.

# # Question 4 

# In[367]:


data =pd.read_csv('C:/Users/renz/Desktop/2힉기/CRM/mailorder.csv', engine='python') 
train = data.iloc[ 0 : 2000, ]
test = data.iloc[ 2000 : ] 

lin = LinearRegression()
lin.fit(train[['recency','frequency','monetary']], train[['purchase']])
proba = lin.predict(test[['recency','frequency','monetary']])

test['purchase_predicted'] = proba
val500 = test.sort_values(by = 'purchase_predicted', ascending = False).reset_index().iloc[0:500 ,]
print('The number of customers who actually made purchases :', val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 2x2x2 RFM codes를 적용했을 때의 78명 , 5x5x5 RFM codes를 적용했을 때의 63명과 비교하면
# 
# 회귀분석을 시행하였을때는 실제구매한 사람의 수가 80명으로 어느정도 향상되었다고 볼 수 있습니다.

# # Question 5

# In[418]:


from sklearn.ensemble import RandomForestRegressor
ran = RandomForestRegressor(n_estimators = 100, random_state =0)
ran.fit(train[['recency','frequency','monetary']], train[['purchase']])
proba = ran.predict(test[['recency','frequency','monetary']])

test['purchase_predicted'] = proba
val500 = test.sort_values(by = 'purchase_predicted', ascending = False).reset_index().iloc[0:500 ,]
print('by 랜덤 포레스트 회귀')
print('The number of customers who actually made purchases :', val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 랜덤포레스트 회귀분석을 이용하였고 회귀분석시 트리 100개를 앙상블하여 학습을 시켰으나 앞의 모델과 비교하여 
# 
# 성능이 전혀 개선되지 않았습니다.

# In[419]:


from sklearn.linear_model import Ridge
rid = Ridge(alpha = 10)
rid.fit(train[['recency','frequency','monetary']], train[['purchase']])
proba = rid.predict(test[['recency','frequency','monetary']])

test['purchase_predicted'] = proba
val500 = test.sort_values(by = 'purchase_predicted', ascending = False).reset_index().iloc[0:500 ,]
print('by 릿지 회귀')
print('The number of customers who actually made purchases :',val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 변수에 대한 규제를 가하기 위하여 규제강도를 10으로써 놓고 학습을 시켰으나 일반적인 선형회귀 분석에 비하여 
# 
# 개선이 되지 않았습니다.

# In[420]:


from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100,500), solver ='adam', random_state =0)
mlp.fit(train[['recency','frequency','monetary']], train[['purchase']])
proba = mlp.predict(test[['recency','frequency','monetary']])

test['purchase_predicted'] = proba
val500 = test.sort_values(by = 'purchase_predicted', ascending = False).reset_index().iloc[0:500 ,]
print('by 다층퍼셉트론 회귀')
print('The number of customers who actually made purchases :',val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 다층 퍼셉트론 회귀분석으로 hidden layer를 2개로 각각의 노드수를 100, 500으로 했지만 다른 모델에 비하여 
# 
# 개선이 되지 않았습니다.

# In[421]:


data =pd.read_csv('C:/Users/renz/Desktop/2힉기/CRM/mailorder.csv', engine='python') 
train = data.iloc[ 0 : 2000, ]
test = data.iloc[ 2000 : ] 

train = train[ (train['purchase'] == 0) | ((train['purchase'] == 1) & (train['recency'] < 25))] 

train = pd.get_dummies(train)
test= pd.get_dummies(test)

train['sq_recency'] = pow(train['recency'],2)
test['sq_recency'] = pow(test['recency'],2)

lin = LinearRegression()

lin.fit(train[['frequency','recency','sq_recency','monetary','gender_F','gender_M']], train[['purchase']])
proba = lin.predict(test[['frequency','recency','sq_recency','monetary','gender_F','gender_M']])

test['purchase_predicted'] = proba
val500 = test.sort_values(by = 'purchase_predicted', ascending = False).reset_index().iloc[0:500 ,]

print('by 일반적인 선형회귀')
print('The number of customers who actually made purchases :',val500['purchase'].sum())
print('ratio :', val500['purchase'].sum()/500)


# 앞선 모델들과는 다르게 성별변수를 더미화시켜서 집어넣었고 또한, 아래의 boxplot에서 보시는 봐와 같이 recency의 값이
# 
# 25 이상인 이상값들을 제거하여 test의 실제 물건을 구매한 사람을 예측하는데 있어서 영향을 받지 않도록 하였습니다.
# 
# 그러고 나서 recency의 값들을 제곱하여 새로운 변수로서 추가를 하였습니다. 그 결과 실제로 구매한 사람들의 수가 96명으로 
# 
# 집계되어 앞선 모델들보다 확연한 성능의 향상을 볼 수 있었습니다.  
# 
# 일반적인 선형회귀 뿐만 아니라 라소회귀, 릿지회귀, 랜덤포레스트 회귀를 같은 변수를 가지고 학습을 시행 하였지만 96명 이상
# 
# 으로 나오지는 않았습니다.

# In[417]:


data =pd.read_csv('C:/Users/renz/Desktop/2힉기/CRM/mailorder.csv', engine='python') 
train = data.iloc[ 0 : 2000, ]
test = data.iloc[ 2000 : ] 

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.xlabel('train')

train[train['purchase'] == 1].boxplot(['recency'])
plt.subplot(1,2,2)
plt.xlabel('test')
test[test['purchase'] == 1].boxplot(['recency'])

