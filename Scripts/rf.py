import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import copy

def convert_to_lower(df,column):
   df[column]= df[column].astype(str).str.lower()
   return df

def one_hot_encoding(df,column):
   df[column] = df[column].str.strip()
   df[column] = df[column].str.lower()
   df[column] = df[column].astype('category')
   category_list = sorted(list(df[column].unique()))
   df[column] = df[column].cat.codes
   df = pd.get_dummies(df,prefix=[column],columns=[column])
   df.drop(columns=df.columns[-1], axis=1, inplace=True)
   return df

if __name__ == "__main__":
  df = pd.read_csv("success_data.csv")
  df.drop('company_name', axis=1, inplace=True)
  df = convert_to_lower(df,'category_list')
  df = convert_to_lower(df,'status')
  df = one_hot_encoding(df,'category_list')
  df = one_hot_encoding(df,'status')
  print("done 1")
  df["target_variable"] = df["target_variable"].astype(int)
  #df.to_csv('converted_success_data.csv',index=False)
  #df = pd.read_csv("converted_success_data.csv")
  y = df['target_variable']
  df.drop('target_variable', axis=1, inplace=True)

  ############ Starting loops ############
  accuracies = []
  precisions = []
  recalls = []
  for i in range(5):
    print("Iteration : ",i)
    X = copy.deepcopy(df)
    rs = random.randint(100,200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rs)
    # model = LogisticRegressionCV(cv=5,random_state=0,verbose=0)
    model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt',verbose=0,max_depth=40)
    model.fit(X_train, y_train)
    print("done 2")
    y_pred = pd.Series(model.predict(X_test))
    y_test = y_test.reset_index(drop=True)
    z = pd.concat([y_test, y_pred], axis=1)
    z.columns = ['True', 'Prediction']
    z.head()
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
  print(accuracies)
  print(precisions)
  print(recalls)
  
  # grid search
  # cross validation
  # dropping columns
  # 