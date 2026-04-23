import time
import os, sys
import pickle
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

project = sys.argv[1]
# proj_bug_cnt = len(pickle.load(open(project + '.pkl', 'rb')))

seed = 0
batch_size = 60
learning_rate = 1e-2

# Chart-21

# cmd = f"python grace_train_new.py Chart-1 Chart {learning_rate} {seed} {batch_size}"
cmd = f"python grace_train.py {project}-1 {project} {learning_rate} {seed} {batch_size}"
# cmd = f"python grace_train.py Math-1 Math {learning_rate} {seed} {batch_size}"
# cmd = f"python grace_train_new.py Math2 {project} {learning_rate} {seed} {batch_size}"def
subprocess.run(cmd, shell=True)

# for bug_id in range(proj_bug_cnt):
#     cmd = f"python grace_train.py {bug_id} {project} {learning_rate} {seed} {batch_size}"
#     subprocess.run(cmd, shell=True)
    
# cmd_sum = f"python sum.py {project} {seed} {learning_rate} {batch_size}"
# subprocess.run(cmd_sum, shell=True)

# cmd_watch = f"python watch.py {project} {seed} {learning_rate} {batch_size}"
# subprocess.run(cmd_watch, shell=True)

# def merge_fl_result(result_path):
#     t = {}
#     for i in range(0, versionNum[proj]):
#         if not os.path.exists(result_path + proj + 'res%d_%d_%s_%s.pkl'%(i, seed, lr, batch_size)):
#             continue
#         p = pickle.load(open(result_path + proj + 'res%d_%d_%s_%s.pkl'%(i,seed, lr, batch_size), 'rb'))
#         # print(p)
#         for k, v in p.items():
#             print(k)
#             print(v)
#             print('='*100)
#         sys.exit(0)
#         for x in p:
#             t[x] = p[x]