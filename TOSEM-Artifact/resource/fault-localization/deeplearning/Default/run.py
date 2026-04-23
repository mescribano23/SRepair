import re
import time
import json
import os, sys
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
POOL_SIZE_DICT = {'Time':10, 'Math':20, "Lang":20, "Chart":6, "Mockito":8, "Closure":2}

RESULT_DIR = '/data/FL/GRACE/Grace/Default/result'

def run_task(index, project, lr, seed, batch_size):
    cmd = f"python grace_train.py {index} {project} {lr} {seed} {batch_size}"
    subprocess.run(cmd, shell=True)
    time.sleep(10)


if __name__ == "__main__":
    lr = 1e-2
    seed = 0
    batch_size = 60

    project = sys.argv[1]
    pool_size = 10
    with open(f'/data/FL/GRACE/Grace/Data/{project}.json', 'r') as f:
        dataset = json.load(f)

    with Pool(processes=pool_size) as pool:
        tasks = []
        for bug_name in dataset.keys():
            if not dataset[bug_name]['ans']:
                continue
            tasks.append((bug_name, project, lr, seed,batch_size))
        
        pool.starmap(run_task, tasks)


    # merge_result(project)