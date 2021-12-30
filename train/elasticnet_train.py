import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    df_train=pd.read_csv('../datasets/df_train.csv')
    df_test=pd.read_csv('../datasets/df_test.csv')
    print(len(df_train))
    print(len(df_test))
    
    TARGET_COL='Sold_Amount'
    train_x=df_train.drop(TARGET_COL, axis=1)
    train_y=df_train[TARGET_COL]
    test_x=df_test.drop(TARGET_COL, axis=1)
    test_y=df_test[TARGET_COL]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     # Register the model
        #     # There are other ways to use the Model Registry, which depends on the use case,
        #     # please refer to the doc for more information:
        #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #     mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        # else:
        #     mlflow.sklearn.log_model(lr, "model")