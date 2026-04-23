import pickle
import json
import numpy as np
from scipy.sparse import coo_matrix

def default_converter(o):
    if isinstance(o, set):
        return list(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, coo_matrix):
        # 将coo_matrix转换为三元组格式
        return {
            'data': o.data.tolist(),
            'row': o.row.tolist(),
            'col': o.col.tolist(),
            'shape': o.shape
        }
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def pkl_to_json(pkl_file_path, json_file_path):
    # 读取 .pkl 文件
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    
    # 将数据写入 .json 文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2, default=default_converter)

# 示例使用
pkl_file_path = '/data/FL/GRACE/Grace/Data/Chartdata.pkl'  # 你的 .pkl 文件路径
json_file_path = '/data/FL/GRACE/Grace/TMP/Chartdata.json'  # 输出的 .json 文件路径
pkl_to_json(pkl_file_path, json_file_path)