import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json


class SumDataset(data.Dataset):
    def __init__(self, config, dataset_property, proj, val_bug_name, occupied_lst=[]):
        self.proj = proj
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)
        self.Nl_Voc['Test'] = len(self.Nl_Voc)
        self.Nl_Voc['Line'] = len(self.Nl_Voc)
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataset_property
        self.selected_bugs = []
        self.test_bug_name = val_bug_name
        self.bug_name_lst = []
        self.processed_data_path = './temp/' + self.proj + '_processed.pkl'

        if os.path.exists("nl_voc.pkl"):
            self.Load_Voc()
        else:
            self.init_dic()

        # data = self.preProcessData()
        if not os.path.exists(self.processed_data_path):
            # print('[DBEUG] preProcessData')
            self.bug_name_lst, data = self.preProcessData()
        else:
            self.bug_name_lst, data = pickle.load(open(self.processed_data_path, 'rb'))
        self.data = []
        if dataset_property == "train":
            for feature_idx in range(len(data)):
                feature_lst = []
                for curr_bug_name in self.bug_name_lst:
                    if curr_bug_name in occupied_lst:
                        # print('if curr_bug_name in occupied_lst:')
                        # print(curr_bug_name)
                        continue
                    curr_bug_idx = self.bug_name_lst.index(curr_bug_name)
                    feature_lst.append(data[feature_idx][curr_bug_idx])
                self.data.append(feature_lst)

        elif dataset_property == 'test':
            filtered_bugs = [bug for bug in self.bug_name_lst if bug != val_bug_name]
            selected_bug = random.choice(filtered_bugs)
            self.selected_bugs = [selected_bug]
            for feature_idx in range(len(data)):
                feature_lst = []
                selected_bug_idx = self.bug_name_lst.index(selected_bug)
                feature_lst.append(data[feature_idx][selected_bug_idx])
                self.data.append(feature_lst)

        else:
            val_bug_idx = self.bug_name_lst.index(val_bug_name)
            for feature_idx in range(len(data)): 
                feature_lst = []
                feature_lst.append(data[feature_idx][val_bug_idx])
                self.data.append(feature_lst)
            self.selected_bugs = [val_bug_name]

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))

    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
    
    def init_dic(self):
        print("initVoc")
        f = open(self.p + '.pkl', 'rb')
        data = pickle.load(f)
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff = 0)
        self.Code_Voc = code_voc.word2id
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    
    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
        return seq
    
    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans


    def preProcessData(self):
        dataFile = f'../Data/{self.proj}.json'
        
        # bugs = pickle.load(dataFile)
        with open(dataFile, 'r') as f:
            dataset = json.load(f)
        # dataset = json.load(dataFile)
        # print(len(dataset))
        # sys.exit()
        Nodes = []
        LineNodes = []
        Res = []
        inputText = []
        inputNlad = []
        bug_name_lst = []

        for bug_name, bug_data in dataset.items():
                    
            nodes = []
            res = []
            nladrow = []
            nladcol = []
            nladval = []
            texta = []
            textb = []
            linenodes = []
            bug_name_lst.append(bug_name)

            methodnum = len(bug_data['methods'])
            rrdict = {}
            for s in bug_data['methods']:
                rrdict[bug_data['methods'][s]] = s[:s.index('(')]
            for i in range(methodnum):
                nodes.append('Method')
                if len(rrdict[i].split(":")) > 1:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]]) 
                else:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:]) 

                ans = self.splitCamel(tokens)
                ans.remove('.')
                texta.append(ans)
                if i in bug_data['ans']:
                    res.append(1)
                else:
                    res.append(0)

            rrdic = {}
            for s in bug_data['ftest']:
                rrdic[bug_data['ftest'][s]] = s
            for i in range(len(bug_data['ftest'])):
                nodes.append('Test')
                if len(rrdic[i].split(":")) > 1:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                else:
                    tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                ans = self.splitCamel(tokens)
                ans.remove('.')
                textb.append(ans)
            rrdic = {}
            for i in range(len(bug_data['rtest'])):
                nodes.append('RTest')

            for e in bug_data['edge_method2line']:
                a = e[0]
                b = e[1] + self.Nl_Len
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in bug_data['edge_line2rtest']:
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum + len(bug_data['ftest'])
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in bug_data['edge_line2ftest']:
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)

            overlap = self.getoverlap(texta, textb)

            Nodes.append(self.pad_seq(self.Get_Em(nodes, self.Nl_Voc), self.Nl_Len))
            Res.append(self.pad_seq(res, self.Nl_Len))
            inputText.append(self.pad_seq(overlap, self.Nl_Len))
            LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len)) # REMOVE

            row = {}
            col = {}
            for i  in range(len(nladrow)):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range(len(nladrow)):
                nladval[i] = 1 / math.sqrt(row[nladrow[i]]) * 1 / math.sqrt(col[nladcol[i]])
            # print(f'(nladrow, nladcol) {len(nladrow)} {len(nladcol)}')
            nlad = sparse.coo_matrix((nladval, (nladrow, nladcol)), shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len))
            inputNlad.append(nlad)

        batchs = [Nodes, inputNlad, Res, inputText, LineNodes]
        dump_data = (bug_name_lst, batchs)
        # batchs = [Nodes, inputNlad, Res, inputText]
        # self.data = batchs
        # self.bug_name_lst = bug_name_lst
        # print('[DEBUG] batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus]')
        open(self.processed_data_path, "wb").write(pickle.dumps(dump_data, protocol=4))
        return dump_data

    def __getitem__(self, offset):
        ans = []
        for i in range(len(self.data)):
            if i == 2:
                ans.append(self.data[i][offset].toarray())
            else:
                ans.append(np.array(self.data[i][offset]))
        return ans
    
    def __len__(self):
        return len(self.data[0])
    
    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        # print('[debug] Get_Train(self, batch_size):')
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            for i in range(batch_nums):
                ans = []
                for j in range(len(data)):
                    if j != 1:  
                    # if not isinstance(data[j], torch.Tensor): 
                        # print(self.dataName)
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        # print(j)
                        # print(self.dataName)
                        # print(data[j])
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse_coo_tensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                        # ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 1:
                    # if not isinstance(data[j], torch.Tensor): 
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size: ]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse_coo_tensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                        # ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            