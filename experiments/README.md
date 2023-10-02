The results in the paper can be reproduced by running the following scripts, where RESULT_PATH is the path to the directory to store the results.

To run 100 experiments with each of the 6 DGPs:
python3 exp_toy.py -u 0.5 -p RESULT_PATH

To run 100 experiments on all semi-synthetic datasets except for the criteo datasets: 
python3 exp_all.py -u 0.5 -p RESULT_PATH

Since each instance of experiment on criteo dataset takes longer to run, they are ran with a job array with slurm that runs 100 jobs concurrently.
To run a single experiment on the criteo dataset with simple CATE:
python3 exp_single.py -p results_nu_05/criteo_true_${SLURM_ARRAY_TASK_ID}.jbl -r ${SLURM_ARRAY_TASK_ID} -u 0.5 -s
To run a single experiment on the criteo dataset with fitted CATE:
python3 exp_single.py -p results_nu_05/criteo_false_${SLURM_ARRAY_TASK_ID}.jbl -r ${SLURM_ARRAY_TASK_ID} -u 0.5


Tables and Figures can be generated using Analysis.ipynb.
