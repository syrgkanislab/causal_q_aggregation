from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from aggregation import experiment
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed 
import warnings
from joblib import effective_n_jobs
import argparse
import time
import multiprocessing
import os
                              

print(f"Number of CPUs =  {multiprocessing.cpu_count()}")


parser = argparse.ArgumentParser()
parser.add_argument("-u","--nu",type = float)
parser.add_argument("-p","--path")
args = parser.parse_args()


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

all_results_full = {}
tic = time.perf_counter()
for ds in ['401k', 'welfare', 'star', 'poverty']:
    for simple in [True, False]:
        all_results_full[(ds, simple)] = Parallel(n_jobs=-1, verbose=10)(delayed(experiment)(ds,
                                                             semi_synth=semi_synth,
                                                             simple_synth=simple,
                                                             max_depth=max_depth,
                                                             scale=scale,
                                                             true_f=simple_true_cef,
                                                             random_state=it,
                                                             nu = args.nu)
                                                          for it in range(100))
        joblib.dump(all_results_full,os.path.join(args.path, 'semi_synthetic.jbl'))
        
        toc = time.perf_counter()        
        print(f"Runtime =  {toc - tic:0.4f} seconds")
        
        
        
        
        
        
        
