import pandas as pd
import numpy as np
import scipy.io
import random
import pickle
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


class meta_dataset(Dataset):
    def __init__(
        self, data, x_length, y_length, data_num, mean, std, three_sigma=False
    ):
        self.data = data
        self.x_len = x_length
        self.y_len = y_length
        self.data_num = data_num
        self.mean = mean
        self.std = std
        self.three_sigma = three_sigma

    def __len__(self):
        return self.data_num

    def normalization(self, data):
        return (data - self.mean) / self.std

    def three_sigma_func(self, data):
        data[data > 3.0] = 3.0
        data[data < -3.0] = -3.0
        return data

    def __getitem__(self, index):
        data = self.data[index : index + self.x_len + self.y_len]
        X = data[: self.x_len]
        y = data[self.x_len :]

        if self.std != 0:
            X = self.normalization(X)
            y_scaled = self.normalization(y)
            if self.three_sigma:
                X = self.three_sigma_func(X)
                y_scaled = self.three_sigma_func(y_scaled)
        else:
            y_scaled = y

        X = torch.FloatTensor(X)
        ori_y = torch.FloatTensor(y)
        y_scaled = torch.FloatTensor(y_scaled)
        if X.dim() < 2:
            X = X.unsqueeze(0)

        return X, y_scaled, ori_y


def process_data(
    data, x_length=2 * 144, y_length=1, test_rate=0.2, query_rate=0.1, three_sigma=False
):
    data = data.astype(float)
    test_size = int(len(data) * test_rate)
    train_size = len(data) - test_size

    train_data = data[:train_size]
    mean, std = train_data.mean(), train_data.std()
    std += 1e-8
    train_data_num = len(train_data) - x_length - y_length + 1
    train_dataset = meta_dataset(
        data=train_data,
        x_length=x_length,
        y_length=y_length,
        data_num=train_data_num,
        mean=mean,
        std=std,
        three_sigma=three_sigma,
    )

    test_data = data[train_size:]
    test_data_num = len(test_data) - x_length - y_length + 1
    test_dataset = meta_dataset(
        data=test_data,
        x_length=x_length,
        y_length=y_length,
        data_num=test_data_num,
        mean=mean,
        std=std,
        three_sigma=three_sigma,
    )

    any_minus = (
        (train_data_num <= 0)
        # or (supp_data_num <= 0)
        # or (query_data_num <= 0)
        or (test_data_num <= 0)
    )

    data_dict = {"train": train_dataset, "test": test_dataset, "mean": mean, "std": std}

    return data_dict, any_minus


def GuangzhouData(x_length=2 * 144, y_length=1, query_rate=0.1, test_rate=0.2):
    if os.path.isfile("Guangzhou/tasks.pickle"):
        with open("Guangzhou/tasks.pickle", "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_csv("Guangzhou/traffic_speed_data.csv")
        road_data = df.groupby(["road_id"])
        data_list = []

        for i in range(1, 215):
            data = road_data.get_group(i)["speed"].values
            data_dict, any_minus = process_data(data, x_length, y_length, query_rate)
            if any_minus:
                continue
            data_list.append(data_dict)

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open("Guangzhou/tasks.pickle", "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def GuangzhouDataShort(
    x_length=2 * 144, y_length=1, query_rate=0.2, test_rate=0.2, total_length=None
):
    file_name = "Guangzhou/tasks-{}.pickle".format(total_length)
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_csv("Guangzhou/traffic_speed_data.csv")
        road_data = df.groupby(["road_id"])
        data_list = []

        for i in range(1, 215):
            data = road_data.get_group(i)["speed"].values
            if total_length is not None:
                total_length = min(len(data), total_length)
            else:
                total_length = len(data)
            data = data[:total_length]
            data_dict, any_minus = process_data(data, x_length, y_length, query_rate)
            if any_minus:
                continue
            data_list.append(data_dict)

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open(file_name, "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def SeattleData(x_length=2 * 144, y_length=1, query_rate=0.1, test_rate=0.2):
    if os.path.isfile("Seattle/tasks.pickle"):
        with open("Seattle/tasks.pickle", "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_pickle("Seattle/speed_matrix_2015")
        start_stamp = "2015-01-01 00:00:00"
        end_stamp = "2015-01-28 23:55:00"
        select_data = df.loc[
            start_stamp:end_stamp
        ].values  # time_stamps x number of mileposts
        data_list = []

        for i in range(select_data.shape[1]):
            data = select_data[:, i]
            data_dict, any_minus = process_data(data, x_length, y_length, query_rate)
            if any_minus:
                continue
            data_list.append(data_dict)

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open("Seattle/tasks.pickle", "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def SeattleDataShort(
    x_length=2 * 144, y_length=1, query_rate=0.2, test_rate=0.2, total_length=None
):
    file_name = "Seattle/tasks-{}.pickle".format(total_length)
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_pickle("Seattle/speed_matrix_2015")
        start_stamp = "2015-01-01 00:00:00"
        end_stamp = "2015-01-28 23:55:00"
        select_data = df.loc[
            start_stamp:end_stamp
        ].values  # time_stamps x number of mileposts
        data_list = []

        for i in range(select_data.shape[1]):
            data = select_data[:, i]
            if total_length is not None:
                total_length = min(len(data), total_length)
            else:
                total_length = len(data)
            data = data[:total_length]
            data_dict, any_minus = process_data(data, x_length, y_length, query_rate)
            if any_minus:
                continue
            data_list.append(data_dict)

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open(file_name, "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def FedCSISData(
    x_length=7 * 24,
    y_length=1,
    query_rate=0.2,
    test_rate=0.2,
    num_device=1000,
    column="Mean",
):
    series_list = [
        "memoory_free",
        "cpu_usage",
        "memoryallocatedbyproc",
        "in_traffic",
        "out_traffic",
    ]
    file_name = "FedCSIS/tasks-{}-{}.pickle".format(num_device, column)
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_csv("FedCSIS/training_series_long.csv")
        device_list = []
        for k1, k2 in df.groupby(by=["hostname", "series"]):

            if k1 not in device_list:
                device_list.append(k1)

        random.shuffle(device_list)
        data_list = []

        for i, device in enumerate(device_list):
            print("Process device {}: {}".format(i + 1, device))
            data = df.loc[(df["hostname"] == device[0]) & (df["series"] == device[1])]
            data["time"] = pd.to_datetime(data["time_window"])
            data.sort_values("time", inplace=True)
            data = data[column]
            data = data.interpolate(method="quadratic")
            data = data.values
            if any(np.isnan(data)):
                continue
            data_dict, any_minus = process_data(
                data, x_length, y_length, query_rate, three_sigma=True
            )
            if any_minus:
                continue
            data_list.append(data_dict)
            print("Already process {} devices".format(len(data_list)))
            if len(data_list) >= num_device:
                break

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open(file_name, "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def FedCSISSeriesData(
    x_length=7 * 24,
    y_length=1,
    query_rate=0.2,
    test_rate=0.2,
    column="Mean",
    series="memoory_free",
):

    file_name = "FedCSIS/tasks-{}-{}.pickle".format(series, column)
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            print("Loading pickle file")
            my_dict = pickle.load(f)
            train_task_list = my_dict["train"]
            test_task_list = my_dict["test"]
    else:
        df = pd.read_csv("FedCSIS/training_series_long.csv")
        device_list = []
        for k1, k2 in df.groupby(by=["hostname", "series"]):
            if (k1 not in device_list) and (k1[1] == series):
                device_list.append(k1)

        random.shuffle(device_list)
        data_list = []

        for i, device in enumerate(device_list):
            print("Process device {}: {}".format(i + 1, device))
            data = df.loc[(df["hostname"] == device[0]) & (df["series"] == device[1])]
            data["time"] = pd.to_datetime(data["time_window"])
            data.sort_values("time", inplace=True)
            data = data[column]
            data = data.interpolate(method="quadratic")
            data = data.values
            if any(np.isnan(data)):
                continue
            data_dict, any_minus = process_data(
                data, x_length, y_length, query_rate, three_sigma=True
            )
            if any_minus:
                continue
            data_list.append(data_dict)
            print("Already process {} devices".format(len(data_list)))

        random.shuffle(data_list)
        test_size = int(len(data_list) * test_rate)
        train_size = len(data_list) - test_size
        train_task_list = data_list[:train_size]
        test_task_list = data_list[train_size:]

        with open(file_name, "wb") as f:
            my_dict = {"train": train_task_list, "test": test_task_list}
            pickle.dump(my_dict, f)

    return train_task_list, test_task_list


def main():
    train_task_list, test_task_list = FedCSISData(x_length=7 * 24, y_length=1)


if __name__ == "__main__":
    train_task_list, test_task_list = FedCSISSeriesData()
