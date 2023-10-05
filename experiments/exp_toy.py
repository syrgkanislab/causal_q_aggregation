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

all_results = {}
tic = time.perf_counter()
for n in [1000, 10000]:
    for dgp in np.arange(1, 7):
        all_results = Parallel(n_jobs=-1, verbose=3)(delayed(experiment)(dgp, n=n,
                                                                              random_state=it,
                                                                              scale=0.1,
                                                                              nu=args.nu)
                                                          for it in range(100))
        joblib.dump(all_results,os.path.join(args.path, f'results_dgp_{dgp}_n_{n}_scale_01.jbl'))
        toc = time.perf_counter()        
        print(f"Runtime =  {toc - tic:0.4f} seconds")
