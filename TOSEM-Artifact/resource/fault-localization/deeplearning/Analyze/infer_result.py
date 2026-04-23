import re
import json
from pathlib import Path

def merge_result():
    proj_result_path = Path(RESULT_DIR) / project_infer
    print(proj_result_path)
    pattern = re.compile(rf'^{project}-\d+$')
    # pattern = re.compile(rf'^{project}\d+$')
    # pattern = re.compile(rf'^{project}_\d+$')
    
    # merge_result = {}
    top1 = 0
    top3 = 0
    top5 = 0
    cnt = 0
    for file_path in proj_result_path.rglob('*'):
        # print(file_path)
        if file_path.is_file() and pattern.match(file_path.stem):
            cnt += 1
            with open(file_path, 'rb') as f:
                curr_result = json.load(f)
                if curr_result['infer_ans_idx'] <= 4:
                    top5 += 1
                    if curr_result['infer_ans_idx'] <= 2:
                        top3 += 1
                        if curr_result['infer_ans_idx'] == 0:
                            top1 += 1
    print(f'TOP-1: {top1}')
    print(f'TOP-3: {top3}')
    print(f'TOP-5: {top5}')
    print(f'ALL: {cnt}')
                # for bug_id in curr_result:
                #     merge_result[bug_id] = curr_result[bug_id]
    
    # merge_result_file_path = proj_result_path / f'{project}_merge.pkl'


RESULT_DIR = '/data/FL/GRACE/Grace/Default/result'
# project = 'Math'
project = 'Closure'
project_infer = 'Closure_infer'
# project_infer = 'Math_infer'
merge_result()
