#!/usr/bin/env bash
#SBATCH --array=0-99
#SBATCH -n 1
#SBATCH --mem=24GB
#SBATCH --time=10:00:00
#SBATCH --job-name=c_t
#SBATCH --output=logs/Array-%A.%a.out


# ml python/3.6.1
# ml py-numpy/1.19.2_py36
# ml py-scipy/1.4.1_py36
# ml py-scikit-learn/0.24.2_py36
# ml py-pandas/1.0.3_py36

ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scipy/1.6.3_py39
ml py-scikit-learn/1.0.2_py39 
ml py-pandas/2.0.1_py39


export PYTHONPATH=$GROUP_HOME/python_env/causal_q_agg_2/lib/python3.9/site-packages:$PYTHONPATH

# for i in {0..9}; do
#     srun -n 1 python3 exp_xg_single.py -p results_criteo/criteo_true_$((SLURM_ARRAY_TASK_ID+i)).jbl -r $((SLURM_ARRAY_TASK_ID+i)) -s &
#     srun -n 1 python3 exp_xg_single.py -p results_criteo/criteo_false_$((SLURM_ARRAY_TASK_ID+i)).jbl -r $((SLURM_ARRAY_TASK_ID+i)) &
# done

# wait

# simple mode
python3 exp_single.py -p results_nu_05/criteo_true_${SLURM_ARRAY_TASK_ID}.jbl -r ${SLURM_ARRAY_TASK_ID} -u 0.5 -s

# fitted mode
# python3 exp_single.py -p results_nu_05/criteo_false_${SLURM_ARRAY_TASK_ID}.jbl -r ${SLURM_ARRAY_TASK_ID} -u 0.5