from pprint import pprint

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.xgboost
import sys

# from utils import fetch_logged_data
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

def main():
    # prepare example dataset
    df_train=pd.read_csv('../datasets/df_train.csv')
    df_test=pd.read_csv('../datasets/df_test.csv')
    print(len(df_train))
    print(len(df_test))
    
    TARGET_COL='Sold_Amount'
    x_train=df_train.drop(TARGET_COL, axis=1)
    y_train=df_train[TARGET_COL]
    x_test=df_test.drop(TARGET_COL, axis=1)
    y_test=df_test[TARGET_COL]

    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    with mlflow.start_run():
        # n_estimators=20
        # max_depth=3
        reg_lambda=1
        gamma=0
        
        n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        regressor = xgb.XGBRegressor(n_estimators=n_estimators, reg_lambda=reg_lambda, gamma=gamma, max_depth=max_depth)
        regressor.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        y_pred = regressor.predict(x_test)

        (rmse, mae) = eval_metrics(y_test, y_pred)

        print("XGBoost model (n_estimators={}, max_depth={}):" .format(n_estimators, max_depth))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)

        mlflow.log_param("n_estimators", n_estimators)
        # mlflow.log_param("reg_lambda", reg_lambda)
        # mlflow.log_param("gamma", gamma)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

    # show logged data
    # for key, data in fetch_logged_data(run.info.run_id).items():
    #     print("\n---------- logged {} ----------".format(key))
    #     pprint(data)


if __name__ == "__main__":
    main()
