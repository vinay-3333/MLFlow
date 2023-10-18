import pandas as pd
import numpy as np
import os 

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

import argparse



def get_data():
    URL ="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    try:
        df = pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e


def evaluate(y_test,y_pred,pred_pro):
    '''mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)
'''
    accuracy = accuracy_score(y_test,y_pred)
    rc_score = roc_auc_score(y_test,pred_pro,multi_class ='ovr')
    # return mae,mse,rmse,r2
    return accuracy,rc_score

def main(n_estimators,max_depth):
    # train test split with raw data
    df = get_data()
    train,test = train_test_split(df)
    X_train=train.drop(['quality'],axis=1)
    X_test = test.drop(['quality'],axis=1)

    y_train = train[['quality']]
    y_test = test[['quality']]

    '''# model training
    lr = ElasticNet()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)'''


    # model training 
    with mlflow.start_run():
        lr = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)


        pred_pro = lr.predict_proba(X_test)

        # evalued the model
        # mae,mse,rmse,r2 = evaluate(y_test,y_pred)
        
        accuracy,rc_score= evaluate(y_test,y_pred,pred_pro)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param('max_depth',max_depth)

        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('roc_auc_score',rc_score)

        # mlflow model logging
        mlflow.sklearn.log_model(lr,"randomforestmodel")

        print(f"accuracy {accuracy}")
        print(f"roc_auc_score {rc_score}")



    # print(f"mean absolute error {mae}, mean squared error {mse}, root mean squared error {rmse}, r2 score {r2}")
    
   
    
if __name__ =='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50,type=int)
    args.add_argument("--max_depth","-m",default=5,type=int)
    parse_args = args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e