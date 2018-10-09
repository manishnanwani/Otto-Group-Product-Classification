import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
os.chdir("F:\\Hackathon\\4. Otto group Classification")

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df1 = df.copy()

df.describe()
pd.options.display.max_rows = 150
df.info()
df.isnull().sum()
test.isnull().sum()

def rstr(df): return df.apply(lambda x: [x.unique()])

print(rstr(df))

range(len(df.columns))

r = list()
c = list()
cor = list()
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        if((df.iloc[:,i].corr(df.iloc[:,j]) > 0.70 or df.iloc[:,i].corr(df.iloc[:,j]) < -0.70) and df.iloc[:,i].corr(df.iloc[:,j])!=1):
            r.append(df.columns[i])
            c.append(df.columns[j])
            cor.append(df.iloc[:,i].corr(df.iloc[:,j]))

cor_df = pd.DataFrame({"1":r, "2":c, "corr value" : cor})
df.drop(['feat_46', 'feat_64', 'feat_72', 'feat_84', 'feat_45'],axis = 1,inplace = True)
test.drop(['feat_46', 'feat_64', 'feat_72', 'feat_84', 'feat_45'],axis = 1,inplace = True)

df_x = df.drop(['target','id'], axis = 1)
df_y = df['target']
final_id = test.id
test.drop('id', axis = 1,inplace = True)

## Using RFE
rfe = RFE(model)
rfe = rfe.fit(df_x, df_y)
df_x_rfe = df_x.loc[:,rfe.support_]
test_rfe = test.loc[:,rfe.support_]

## Logistic Regression
model = LogisticRegression()
model.fit(df_x,df_y)


result = model.predict_proba(test)
accuracy = accuracy_score(df_y, model.predict(df_x))

print(accuracy)

result_lr = pd.DataFrame({'id': final_id,'Class_1': result[:,0],'Class_2': result[:,1],'Class_3': result[:,2],
                          'Class_4': result[:,3],'Class_5': result[:,4],'Class_6': result[:,5],'Class_7': result[:,6],
                          'Class_8': result[:,7],'Class_9': result[:,8]})

result_lr = result_lr[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]

# OneHotEncoder is used for integer values, pd.get_dummies is used for string values
#one_hot = pd.get_dummies(result_lr.result)
#result_lr.drop('result', axis = 1, inplace = True)
#result_lr = result_lr.join(one_hot)

result_lr.to_csv('submission_lr.csv', index = False)


## RandomForest
modelrf = RandomForestClassifier()
modelrf.fit(df_x,df_y)

result = modelrf.predict_proba(test)
accuracy = accuracy_score(df_y, modelrf.predict(df_x))

print(accuracy)

result_rf = pd.DataFrame({'id': final_id,'Class_1': result[:,0],'Class_2': result[:,1],'Class_3': result[:,2],
                          'Class_4': result[:,3],'Class_5': result[:,4],'Class_6': result[:,5],'Class_7': result[:,6],
                          'Class_8': result[:,7],'Class_9': result[:,8]})

result_rf = result_lr[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]

result_rf.to_csv('submission_rf.csv', index = False)

imp = pd.DataFrame({'col': df_x.columns, 'imp':modelrf.feature_importances_})
imp1 = imp.sort_values(by = 'imp', ascending = False)
imp1.index = range(88)
cols = imp1.iloc[0:46,0]
df_x_imp = df_x[cols]
test_imp = test[cols]

## Logistic Reg after keeping only the imp features from Random Forest

model = LogisticRegression()
model.fit(df_x_imp,df_y)


result = model.predict_proba(test_imp)
accuracy = accuracy_score(df_y, model.predict(df_x_imp))

print(accuracy)

result_lr_imp = pd.DataFrame({'id': final_id,'Class_1': result[:,0],'Class_2': result[:,1],'Class_3': result[:,2],
                          'Class_4': result[:,3],'Class_5': result[:,4],'Class_6': result[:,5],'Class_7': result[:,6],
                          'Class_8': result[:,7],'Class_9': result[:,8]})

result_lr_imp = result_lr[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]

result_lr_imp.to_csv('submission_lr_imp.csv', index = False)


## Random Forest after keeping only the imp features from Random Forest
modelrf = RandomForestClassifier()
modelrf.fit(df_x_imp,df_y)

result = modelrf.predict_proba(test_imp)
accuracy = accuracy_score(df_y, modelrf.predict(df_x_imp))

print(accuracy)

result_rf_imp = pd.DataFrame({'id': final_id,'Class_1': result[:,0],'Class_2': result[:,1],'Class_3': result[:,2],
                          'Class_4': result[:,3],'Class_5': result[:,4],'Class_6': result[:,5],'Class_7': result[:,6],
                          'Class_8': result[:,7],'Class_9': result[:,8]})

result_rf_imp = result_lr_imp[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]

result_rf_imp.to_csv('submission_rf_imp.csv', index = False)


## XGBoost
modelxg = XGBClassifier()
modelxg.fit(df_x,df_y)

result = modelxg.predict_proba(test)
accuracy = accuracy_score(df_y, modelxg.predict(df_x))

print(accuracy)

result_xg = pd.DataFrame({'id': final_id,'Class_1': result[:,0],'Class_2': result[:,1],'Class_3': result[:,2],
                          'Class_4': result[:,3],'Class_5': result[:,4],'Class_6': result[:,5],'Class_7': result[:,6],
                          'Class_8': result[:,7],'Class_9': result[:,8]})

result_xg = result_lr[['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']]
result_xg.to_csv('submission_xg.csv', index = False)
