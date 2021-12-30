#!/bin/bash
cd /datium_a/preprocess
python preprocess_1.py
python preprocess_2.py

cd /datium_a/train
python elasticnet_train.py 0 0
python elasticnet_train.py 0.2 0.2
python elasticnet_train.py 0.5 0.5
python xgboost_train.py 5 3
python xgboost_train.py 10 3
python xgboost_train.py 10 5

cd /datium_a/train
mlflow ui --host 0.0.0.0
