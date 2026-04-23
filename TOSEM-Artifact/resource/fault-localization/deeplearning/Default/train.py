import os
import sys
import random
import json
import torch
import pickle
import numpy as np
from Model import *
from tqdm import tqdm
from torch import optim
from Dataset import SumDataset
from ScheduledOptim import ScheduledOptim


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# NlLen_map = {"Time":3900, "Math":4500, "Lang":280, "Chart": 2350, "Mockito":1780, "unknown":2200}
# CodeLen_map = {"Time":1300, "Math":2700, "Lang":300, "Chart":5250, "Mockito":1176, "unknown":2800}
# NlLen_map = {"Time":3900, "Math":45000, "Lang":280, "Chart": 23500, "Mockito":1780, "unknown":2200}
# CodeLen_map = {"Time":1300, "Math":27000, "Lang":300, "Chart": 52500, "Mockito":1176, "unknown":2800}
args = dotdict({
    # 'NlLen':NlLen_map[sys.argv[2]],
    # 'CodeLen':CodeLen_map[sys.argv[2]],
    'NlLen':45000,
    'CodeLen':27000,
    'batch_size':60,
    'embedding_size':32,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'seed':0,
    'lr':1e-3
})

os.environ['PYTHONHASHSEED'] = str(args.seed)

def save_model(model):
    dirs = "/data/FL/GRACE/Grace/Default/model"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model, dirs + f'/Chart_best_model.ckpt')
    # torch.save(model, dirs + f'/Chart_rm_LineNode_edge_best_model.ckpt')


def load_model():
    dirs = "/data/FL/GRACE/Grace/Default/model"
    # assert os.path.exists(dirs + '/Chart_best_model.ckpt'), 'Weights for saved model not found'
    return torch.load(dirs + '/Chart_best_model.ckpt', weights_only=False)
    # return torch.load(dirs + '/Chart_rm_LineNode_edge_best_model.ckpt', weights_only=False)

use_cuda = torch.cuda.is_available()


def gVar(data):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        tensor = [gVar(item) for item in data]
    else:
        assert isinstance(data, torch.Tensor)
        tensor = data

    return tensor.cuda() if use_cuda else tensor

def train(test_bug_name, project):
    random.seed(args.seed+100)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)  
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed) 
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # data = pickle.load(open(project + '.pkl', 'rb'))
    dataFile = f'/data/FL/GRACE/Grace/Data/{project}.json'
        
        # bugs = pickle.load(dataFile)
    with open(dataFile, 'r') as f:
        data = json.load(f)

    # RM DEV
    # dev_set = SumDataset(args, "test", project, val_bug_name=test_bug_name)
    val_set = SumDataset(args, "val", project, val_bug_name=test_bug_name)
    train_set = SumDataset(args, "train", val_bug_name=test_bug_name, proj=project, occupied_lst = val_set.selected_bugs)
    
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)

    model = NlEncoder(args)
    if use_cuda:
        model = model.cuda()

    # rdic = {}
    best_pred_lst = []
    pred_idx_lst = []
    all_epoch_pred_lst = {}

    best_pred_ans_idx = 1e9
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)

    # for x in dev_set.Nl_Voc:
    #   rdic[dev_set.Nl_Voc[x]] = x
    for epoch in tqdm(range(9), desc="Epochs", unit="epoch"):

        validate_flag = False
        for dBatch in train_set.Get_Train(args.batch_size):
            
            # Validate Stage
            if not validate_flag:
                validate_flag = True
                loss = []

                curr_pred_ans_idx = 1e9
                model = model.eval()
                val_set_generator = val_set.Get_Train(len(val_set))
                val_set_batch = next(val_set_generator)

                for i in range(len(val_set_batch)):
                    val_set_batch[i] = gVar(val_set_batch[i])

                with torch.no_grad():
                    _, raw_pred, _ = model(val_set_batch[0], val_set_batch[1], val_set_batch[2], val_set_batch[3], val_set_batch[4])
                    resmask = torch.eq(val_set_batch[0], 2)
                    tmp_pred = -raw_pred
                    tmp_pred = tmp_pred.masked_fill(resmask == 0, 1e9)
                    pred = tmp_pred.argsort(dim=-1)
                    pred = pred.data.cpu().numpy()

                    datat = data[test_bug_name]
                    curr_pred_lst = pred[0].tolist()[:resmask.sum(dim=-1)[0].item()]
                    indices = (curr_pred_lst.index(x) for x in datat['ans'] if x in curr_pred_lst)
                    curr_pred_ans_idx = min(indices, default=1e9)
                    pred_idx_lst.append(curr_pred_ans_idx)
                    # print('curr_pred_lst: ', curr_pred_lst)
                    # print('curr_pred_ans_idx: ', curr_pred_ans_idx)

                all_epoch_pred_lst[epoch] = curr_pred_lst

                if best_pred_ans_idx > curr_pred_ans_idx:
                    best_pred_lst = curr_pred_lst
                    best_pred_ans_idx = curr_pred_ans_idx
                    # save_model(model)
                model = model.train()
            
            # Train Stage
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            loss, _, _ = model(val_set_batch[0], val_set_batch[1], val_set_batch[2], val_set_batch[3], val_set_batch[4])
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()

            optimizer.step_and_update_lr()
    
    print(f'[best_pred_ans_idx]:{best_pred_ans_idx}')
    print(f'[pred_idx_lst]:{pred_idx_lst}')

    pred_fl_result = {
        'bug_id': bug_name,
        'best_pred_ans_idx': best_pred_ans_idx,
        'best_pred_lst': best_pred_lst,
        'pred_idx_lst': pred_idx_lst,
        'all_epoch_pred_lst': all_epoch_pred_lst
    }

    # output_file_path = f'/data/FL/GRACE/Grace/Default/result/{buggy_proj}_train/{buggy_proj}_{bug_id}.json'
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # with open(output_file_path, 'w') as f:
    #     json.dump(pred_fl_result, f, indent=2)



def inference(test_bug_name, project):
    random.seed(args.seed + 100)

    dataFile = f'/data/FL/GRACE/Grace/Data/{project}.json'
    with open(dataFile, 'r') as f:
        data = json.load(f)
    # data = pickle.load(open(project + '.pkl', 'rb'))

    val_set = SumDataset(args, "val", project, test_bug_name)
    
    args.Code_Vocsize = len(val_set.Code_Voc)
    args.Nl_Vocsize = len(val_set.Nl_Voc)
    args.Vocsize = len(val_set.Char_Voc)

    model = NlEncoder(args)
    model = load_model()
    if use_cuda:
        model = model.cuda()
    
    model = model.eval()

    curr_pred_ans_idx = 1e9
    val_set_generator = val_set.Get_Train(len(val_set))
    val_set_batch = next(val_set_generator)

    for i in range(len(val_set_batch)):
        val_set_batch[i] = gVar(val_set_batch[i])

    with torch.no_grad():
        _, raw_pred, _ = model(val_set_batch[0], val_set_batch[1], val_set_batch[2], val_set_batch[3], val_set_batch[4])
        resmask = torch.eq(val_set_batch[0], 2)
        tmp_pred = -raw_pred
        tmp_pred = tmp_pred.masked_fill(resmask == 0, 1e9)
        pred = tmp_pred.argsort(dim=-1)
        pred = pred.data.cpu().numpy()

        datat = data[test_bug_name]
        curr_pred_lst = pred[0].tolist()[:resmask.sum(dim=-1)[0].item()]
        indices = (curr_pred_lst.index(x) for x in datat['ans'] if x in curr_pred_lst)
        curr_pred_ans_idx = min(indices, default=1e9)
        print(f'curr_pred_ans_idx: {curr_pred_ans_idx}/{len(curr_pred_lst)}')

        pred_fl_result = {
            'bug_id': bug_name,
            'infer_ans_idx': curr_pred_ans_idx,
            'infer_pred_lst': curr_pred_lst,
        }

        output_file_path = f'/data/FL/GRACE/Grace/Default/result/{buggy_proj}_infer/{bug_name}.json'
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(pred_fl_result, f, indent=2)
        



if __name__ == "__main__":
    bug_name = sys.argv[1]
    buggy_proj = sys.argv[2]
    args.lr = float(sys.argv[3])
    args.seed = int(sys.argv[4])
    args.batch_size = int(sys.argv[5])
    np.set_printoptions(threshold=sys.maxsize)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # train(bug_name, buggy_proj)

    inference(bug_name, buggy_proj) 

    # best_pred_ans_idx, best_pred_lst, pred_idx_lst, all_epoch_pred_lst = train(bug_id, buggy_proj)




