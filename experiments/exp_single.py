from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from aggregation_XGBoost import experiment
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed 
import warnings
from joblib import effective_n_jobs
import argparse
import time
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument("-p","--path")
parser.add_argument("-r", "--random", type = int)
parser.add_argument("-u", "--nu", type = float)
parser.add_argument("-s", "--simple", action="store_true")

args = parser.parse_args()


print(f"Number of CPUs =  {multiprocessing.cpu_count()}")



print(effective_n_jobs(-1))
warnings.filterwarnings('ignore')
# data = '401k' # which dataset, one of {'401k', 'criteo', 'welfare', 'poverty', 'star'}

## For semi-synthetic data generation
semi_synth = True # Whether true outcome y should be replaced by a fake outcome from a known CEF
# simple_synth = False # Whether the true CEF of the fake y should be simple or fitted from data
max_depth = 2 # max depth of random forest during for semi-synthetic model fitting
scale = .2 # magnitude of noise in semi-synthetic data
def simple_true_cef(D, X): # simple CEF of the outcome for semi-synthetic data
    return .5 * np.array(X)[:, 1] * D + np.array(X)[:, 1]

ds = 'criteo' 
tic = time.perf_counter()
result = experiment(ds,
                 semi_synth=semi_synth,
                 simple_synth=args.simple,
                 max_depth=max_depth,
                 scale=scale,
                 true_f=simple_true_cef,
                 random_state=args.random,
                 nu=args.nu)

joblib.dump(result,args.path )
toc = time.perf_counter()        

print(f"Runtime =  {toc - tic:0.4f} seconds")       
    