import pickle
import json
import sys

def convert_pkl_to_json(pkl_file_path, json_file_path):
    # 从 .pkl 文件中加载数据
    with open(pkl_file_path, 'rb') as file:
        raw_data = pickle.load(file)

    # 递归转换数据中的 set 为 list
    def convert_sets_to_lists(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(item) for item in obj]
        return obj

    # 转换数据中的所有 set
    raw_data = convert_sets_to_lists(raw_data)
    dataset = {}

    for entry in raw_data:
        bug_name = entry["proj"]
        dataset[bug_name] = {}
        dataset[bug_name]['edge_line2ftest'] = entry['edge']
        dataset[bug_name]['edge_method2line'] = entry['edge2']
        dataset[bug_name]['edge_line2rtest'] = entry['edge10']
        dataset[bug_name]['methods'] = entry['methods']
        dataset[bug_name]['ftest'] = entry['ftest']
        dataset[bug_name]['rtest'] = entry['rtest']
        dataset[bug_name]['ans'] = entry['ans']


    print(len(raw_data))
    # for k, v in data.items():
    #     print(k)
    #     print(type(v))

    with open(json_file_path, 'w') as file:
        json.dump(dataset, file, indent=4)


if __name__ == "__main__":
    project = sys.argv[1]
    input_pkl = f'/data/FL/GRACE/Grace/Default/{project}.pkl'
    output_json = f'/data/FL/GRACE/Grace/Data/{project}.json'
    

    convert_pkl_to_json(input_pkl, output_json)
    print(f"Data from {input_pkl} has been successfully converted to {output_json}.")