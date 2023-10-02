#!/usr/bin/env bash
#SBATCH -n 20
#SBATCH --time=24:00:00
#SBATCH --job-name=semi_05
#SBATCH --output=R-%x.%j.out

ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scipy/1.6.3_py39
ml py-scikit-learn/1.0.2_py39 
ml py-pandas/2.0.1_py39


export PYTHONPATH=$GROUP_HOME/python_env/causal_q_agg_2/lib/python3.9/site-packages:$PYTHONPATH

# toy experiments
# python3 exp_toy.py -u 0.5 -p results_nu_05/toy

# all semi-synthetic but criteo
python3 exp_all.py -u 0.5 -p results_nu_05
