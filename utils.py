import numpy as np
import os
import sys
import torch
import importlib
import random

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def read_metadata(meta_path, is_eval=False):
    d_meta = {}
    d_score = {}
    file_list=[]
    with open(meta_path, 'r') as f:
         l_meta = f.readlines()
    
    if (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _,key,_,_,label, quality = line.strip().split()
            quality = float(quality)
            if quality < 2.5:
                d_score[key] = 0
            else:
                d_score[key] = 1
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list, d_score
    
def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def my_collate(batch): #Dataset return sample = (utterance, target, nameFile) #shape of utterance [1, lenAudio]
  data = [dp[0] for dp in batch]
  label = [dp[1] for dp in batch]
  nameFile = [dp[2] for dp in batch]
  return (data, label, nameFile) 
