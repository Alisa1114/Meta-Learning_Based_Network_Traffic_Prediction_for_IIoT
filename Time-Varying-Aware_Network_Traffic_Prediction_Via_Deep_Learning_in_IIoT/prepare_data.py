# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:07:22 2021

@author: Administrator
"""
import numpy as np
import pandas as pd
import random
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def getsingleGroup(pro_data,group,src_len,tar_len,step):
    '''
    pro_data 全部数据
    group 单组 key
    单组生成序列
    
    '''
    current_df=pro_data.loc[(pro_data['hostname']==group[0]) & (pro_data['series']==group[1])]
    tw=src_len+tar_len#总的采样窗口大小，前面是X,后面部分的Mean是Y
    step=step
    train_x = []
    train_y = []
    
        #按时间排序
    current_df['time'] = pd.to_datetime(current_df['time_window'])
    current_df.sort_values('time', inplace=True)
    current_df = current_df.interpolate(method="linear", axis=0).ffill().bfill()
    useful_column=[ 'Mean', 'SD', 'Open', 'High','Low', 'Close', 'Volume']#取特征列
    
    valid_seq = current_df[-tw:][useful_column]
    valid_x = [valid_seq.values[:src_len]]
    valid_y = [valid_seq[-tar_len:]['Mean'].values]
    current_df = current_df[:-tar_len]
    L=len(current_df)
      
    for i in range(0,L-tw,step):
        if i>L-tw and i<L:#处理尾巴上的
            train_seq =current_df[-tw:][useful_column]
            train_x.append(train_seq.values[-tw:tw-src_len])
            train_y.append(train_seq[tw-src_len:]['Mean'].values)
        else:
            train_seq =current_df[i:i+tw][useful_column]
            train_x.append(train_seq.values[:src_len])
            train_y.append(train_seq[-tar_len:]['Mean'].values)
            
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)
    
    return train_x, train_y, valid_x, valid_y

def read_pkl(file_path):
    result_dict = {}
    i = 0
    with open(file_path, 'rb') as file:
        while True:
            try:
                data_dict = pickle.load(file)
                i+=1
                print(i, len(data_dict.keys()))
                # result_dict.update(data_dict)
            except EOFError:
                break
    return result_dict

def get_dataset(inputdir,src_len,tar_len,step=5,sample_pro=10000):
    
    if os.path.exists("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len)):
        pass
        # train_dict=read_pkl("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len))
        # valid_dict=read_pkl("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len))
        # return train_dict, valid_dict
    else:
        pro_data=pd.read_csv(inputdir)
        all_sample=[]
        for k1,k2 in pro_data.groupby(by=['hostname','series']):
            all_sample.append(k1)
        random.shuffle(all_sample) 
        # all_sample=all_sample[:sample_pro]#少搞点试试
        
        train_dict = {}
        valid_dict = {}
        i = 0
        for idx, id_ in enumerate(all_sample):
            device = "{}#{}".format(id_[0], id_[1])
            print("Device {}: {}".format(idx, device))
            train_x, train_y, valid_x, valid_y = getsingleGroup(pro_data, id_, src_len, tar_len, step)
            
            if train_x.size == 0 or train_y.size == 0 or valid_x.size == 0 or valid_y.size == 0:
                continue
            else:
                i += 1
            
            train_dict[device] = (train_x, train_y)
            valid_dict[device] = (valid_x, valid_y)
            
            if (i % 500 == 0) or (i == sample_pro):
                # print(valid_dict)
                with open("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'ab') as f:
                    pickle.dump(train_dict, f)
                with open("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'ab') as f:
                    pickle.dump(valid_dict, f)
                train_dict.clear()
                valid_dict.clear()
                print("Save partial data")
            
            print("Number of processed device: {}".format(i))
            if i >= sample_pro:
                break
        
        print("Total save {} device data".format(i))
        # train_dict=read_pkl("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len))
        # valid_dict=read_pkl("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len))
        # return train_dict, valid_dict
    
if __name__ == "__main__":
    input_dir = "./training_series_long.csv"
    src_len = 24*14
    tar_len = 24*7
    step=1
    sample_pro = 10000
    get_dataset(input_dir, src_len, tar_len, step, sample_pro=sample_pro)
    # print(train_dict)
    # print(valid_dict)