import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import meta_dataset
from dataset import GuangzhouData, SeattleData, FedCSISData
from dataset import GuangzhouDataShort, SeattleDataShort, FedCSISSeriesData
import torch


dataset_name = "FedCSIS"
data_len = 2000
column = "Mean"
series = "cpu_usage"

if dataset_name == "Guangzhou":
    inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
    train_task_list, test_task_list = GuangzhouDataShort(
        x_length=in_len, query_rate=0.2, total_length=data_len
    )
elif dataset_name == "Seattle":
    inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
    train_task_list, test_task_list = SeattleDataShort(
        x_length=in_len, query_rate=0.2, total_length=data_len
    )
elif dataset_name == "FedCSIS":
    inchannel, out_channel, in_len, hidden_size = 1, 1, 168, 64
    train_task_list, test_task_list = FedCSISSeriesData(column=column, series=series)
dir_name = dataset_name + "_plot"
os.makedirs(dir_name, exist_ok=True)
# 設置圖的大小
# plt.figure(figsize=(100, 20))


# 在每個格子裡繪製時間序列
for i, task in enumerate(train_task_list):
    if i >= 10: 
        continue
    
    print("Plot {} time series".format(i + 1))
    train_dataset = task["train"]
    test_dataset = task["test"]
    mean, std = task["mean"], task["std"]

    # plot original data
    train_data = train_dataset.data
    test_data = test_dataset.data

    fig1, ax1 = plt.subplots(figsize=(6, 2))
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    color = np.random.rand(3,)
    ax1.plot(train_data, color=color)
    ax2.plot(test_data, color=color)
    # plt.title("Time Series {}".format(i + 1))
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    # plt.legend()
    ax1.figure.savefig(dir_name + "/supp{}.png".format(i+1))
    ax2.figure.savefig(dir_name + "/qry{}.png".format(i+1))
    # plt.savefig(dir_name + "/time_series_{}.png".format(i + 1))
    plt.close()
