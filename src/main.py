from preprocess import main as run_preprocess
from features import main as run_feature_eng
from train import main as run_train
from predict import main as run_predict
from cli import init_argparser
import os
import pandas as pd
import joblib

ENV = os.environ.get('ENV')

def load_json(path, filename):
    return pd.read_json(path + filename, 
            orient='records', lines=True)

def save_df(df, path, filename):
    df.to_csv(path + filename, index=false)

def load_df(path, filename):
    return pd.read_csv(path + filename) 

def save_splits(save_path, X_train, X_test, y_train, y_test):
    joblib.dump(X_train, save_path + 'X_train.pkl')
    joblib.dump(X_test, save_path + 'X_test.pkl')
    joblib.dump(y_train, save_path + 'y_train.pkl')
    joblib.dump(y_test, save_path + 'y_test.pkl')

def preprocessing(load_path, save_path, save):
    portfolio = load_json(load_path, 'portfolio.json')
    profile = load_json(load_path, 'profile.json')
    transcript = load_json(load_path, 'transcript.json')
    portfolio, profile, offers = run_preprocess(portfolio, profile, transcript)
    if save:
        save_df(offers, save_path, 'offers.csv') 
        save_df(profile, save_path, 'profile.csv') 
        save_df(portfolio, save_path, 'portfolio.csv') 

def features_eng(load_path, save_path, save):
    offers = load_df(load_path, 'offers.csv')
    profile = load_df(load_path, 'profile.csv')
    X_train, X_test, y_train, y_test = run_feature_eng(offers, profile)
    if save:
        save_splits(save_path, X_train, X_test, y_train, y_test)

def training(load_path, save_path, save):
    X_train = joblib.load(load_path + 'X_train.pkl')
    X_test = joblib.load(load_path + 'X_test.pkl')
    y_train = joblib.load(load_path + 'y_train.pkl')
    y_test = joblib.load(load_path + 'y_test.pkl')
    model, S_test, y_test = run_train((X_train, X_test, y_train, y_test))
    if save:
        joblib.dump(model, save_path + 'model.pkl')
        joblib.dump(S_test, save_path + 'S_test.pkl')
        joblib.dump(y_test, save_path + 'y_test.pkl')

def predicting(load_path):
    model = joblib.load(load_path + 'model.pkl')
    S_test = joblib.load(load_path + 'S_test.pkl')
    y_test = joblib.load(load_path + 'y_test.pkl')
    run_predict((model, S_test, y_test))

if ENV == 'PREPROCESS':
    try:
        load_path, save_path, save_flag = init_argparser('preprocess')
        preprocessing(load_path, save_path, save_flag)
    except Exception as e:
        print(e)

if ENV == 'FEATURES':
    try:
        load_path, save_path, save_flag = init_argparser('feature engineering')
        features_eng(load_path, save_path, save_flag)
    except Exception as e:
        print(e)

if ENV == 'TRAIN':
    try:
        load_path, save_path, save_flag = init_argparser('training')
        training(load_path, save_path, save_flag)
    except Exception as e:
        print(e)

if ENV == 'PREDICT':
    try:
        load_path, save_path, _ = init_argparser('predicting')
        predicting(load_path)
    except Exception as e:
        print(e)




