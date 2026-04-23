import pandas as pd
import pickle
import sys

# 指定要加载的 .pkl 文件路径
file_path = '/data/FL/GRACE/Grace/Data/Chart.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

for bug in data:
    for k, v in bug.items():
        tmp = 0
        if k in ['ftest', 'methods', 'rtest']:
            tmp += len(v)
        if tmp != 0:
            print(tmp)
        # if k in ['ftest', 'methods', 'ans', 'rtest', 'lines', 'ltype', 'edge2', 'edge10', 'edge']:
            # print(len(v))
        # print(k)
        # print(type(v))
        
    # sys.exit()

# for sub_data in 
# print(isinstance(data, list))


# print(len(data))

# 如果数据是一个列表，打印前几个元素
# if isinstance(data, list):
#     idx = 8
    # print(type(data))
    # print(data[0])
    # print(type(data[0]))
    # for k, v in data[idx].items():
    #     print('='*100)
    #     print(k)
    #     print(type(k))
    #     print(type(v))
    #     print(v)
    # lst = data[1]
    # for element in lst:
    #     for k, v in element.items():
    #         print(k)
    #         print(v)
    #         print('='  * 100)
    # print("Data is a list. Here are the first few elements:")
    # print(data[:1])  # 显示列表的前5个元素

# 如果数据是一个字典，打印键和值
# elif isinstance(data, dict):
#     print("Data is a dictionary. Here are the keys and a few values:")
#     for key, value in list(data.items())[:1]:  
#         print(f"Key: {key}, Value: {value}")
