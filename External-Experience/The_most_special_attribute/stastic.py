import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from tqdm import tqdm

json_fold = "./Multi-ID-topk/scores-group-Test1-CosFace-r50-topk-1-erase-black/json"

lists = os.listdir(json_fold)

scores = np.zeros(30)

for path in tqdm(lists):
    with open(os.path.join(json_fold,path),'r') as load_f:
        load_dict = json.load(load_f)
    attr_dict = load_dict["attributes-image1"]
    value = np.array(list(attr_dict.values()))[:,-1]

    # value_ = value.copy()
    # value_.sort()
    
    for i in range(30):
        if value[i] == value[i]:
            # scores[i] += value[i] #/ len(lists)
            scores[i] = scores[i] + (1 - np.arccos(value[i]) / math.pi) #/ len(lists)

scores /= len(lists)

np.savetxt('task.txt', scores, fmt="%f", delimiter=" ")
