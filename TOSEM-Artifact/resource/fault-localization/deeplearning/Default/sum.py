import os
import sys
import pickle
versionNum = {'Lang': 65, 'Time': 27, 'Chart': 26, 'Math': 106, 'Closure': 133, 'Mockito': 38,
                   'CommonsCli': 24, 'CommonsCodec': 22, 'CommonsCsv': 12, 'CommonsJXPath': 14,
                   'JacksonCore': 13, 'JacksonDatabind': 39, 'JacksonXml': 5, 'Jsoup': 63}
proj = sys.argv[1]
seed = int(sys.argv[2])
lr = float(sys.argv[3])
batch_size = int(sys.argv[4])

result_path = '/data/FL/GRACE/Grace/Default/result_chart_0830_1350/'

t = {}
for i in range(0, versionNum[proj]):
    if not os.path.exists(result_path + proj + 'res%d_%d_%s_%s.pkl'%(i, seed, lr, batch_size)):
        continue
    p = pickle.load(open(result_path + proj + 'res%d_%d_%s_%s.pkl'%(i,seed, lr, batch_size), 'rb'))
    # print(p)
    # for k, v in p.items():
    #     print(k)
    #     print(v)
    #     print('='*100)
    # sys.exit(0)
    for x in p:
        t[x] = p[x]


    
open(result_path + proj + 'res_%d_%s_%s.pkl'%(seed,lr, batch_size), 'wb').write(pickle.dumps(t))
print(len(t))
