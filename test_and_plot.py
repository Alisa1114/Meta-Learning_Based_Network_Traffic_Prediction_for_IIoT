import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)
import argparse
from model import get_model
from dataset import meta_dataset
from dataset import GuangzhouData, SeattleData, FedCSISData
from dataset import GuangzhouDataShort, SeattleDataShort, FedCSISSeriesData
from utils import plot_result, reverse_scale
from test import test

def get_parser():
    parser = argparse.ArgumentParser(description='test setting')
    parser.add_argument('-m', '--model_type', default='attn', type=str) # cnn, lstm, attn
    parser.add_argument('-d', '--dataset', default='FedCSIS', type=str)
    parser.add_argument('-c', '--column', default='Mean', type=str)
    # series_list = [
    # "cpu_usage",
    #     "memoory_free",
    #     "memoryallocatedbyproc",
    #     "in_traffic",
    #     "out_traffic",
    # ]
    parser.add_argument('-s', '--series', default='cpu_usage', type=str)
    return parser

def train(
    training,
    test_task_list,
    epochs,
    device,
    dataset_name,
    inchannel,
    out_channel,
    in_len,
    hidden_size,
    column=None,
    model_type="cnn",
    series=None,
):
    result_list = []

    for i, task in enumerate(test_task_list):
        test_result = {"mape": [], "rmse": [], "r2": [], "mae": []}
        train_dataset = task["train"]
        test_dataset = task["test"]
        mean, std = task["mean"], task["std"]
        data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        model = get_model(
            dataset_name, inchannel, out_channel, in_len, hidden_size, model_type
        )
        model = model.to(device)
        if training == "maml":
            if dataset_name == "FedCSIS":
                model.load_state_dict(
                    torch.load(
                        "weights/{}-{}-{}-{}.pth".format(
                            dataset_name, series, column, model_type
                        )
                    )
                )
            else:
                model.load_state_dict(
                    torch.load("weights/{}-{}-2000.pth".format(dataset_name, model_type))
                )
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for X, y, _ in data_loader:
                X = X.to(device)
                y = y.to(device)

                output = model(X)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(
                "Dataset: {} Epoch: {} Loss: {}".format(
                    i + 1, epoch + 1, total_loss / len(data_loader)
                )
            )

            model.eval()
            mape, rmse, r2, mae = test(
                test_dataset,
                model,
                device,
                mean,
                std,
                dataset_name,
                i + 1,
                False,
            )
            result = (
                "Dataset: {} MAPE: {:.4f} RMSE: {:.4f} R2: {:.4f} MAE: {:.4f}".format(
                    i + 1, mape, rmse, r2, mae
                )
            )
            print(result)
            test_result["mape"].append(mape)
            test_result["rmse"].append(rmse)
            test_result["r2"].append(r2)
            test_result["mae"].append(mae)

        result_list.append(test_result)
    return result_list


def main(argss):
    model_type = argss.model_type
    dataset_name = argss.dataset
    column = argss.column
    series = argss.series

    if dataset_name == "Guangzhou":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
        # train_task_list, test_task_list = GuangzhouData(
        #     x_length=in_len, y_length=out_channel
        # )
        train_task_list, test_task_list = GuangzhouDataShort(
            x_length=in_len, query_rate=0.2, total_length=2000
        )

    elif dataset_name == "Seattle":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
        # train_task_list, test_task_list = SeattleData(
        #     x_length=in_len, y_length=out_channel
        # )
        train_task_list, test_task_list = SeattleDataShort(
            x_length=in_len, query_rate=0.2, total_length=2000
        )

    elif dataset_name == "FedCSIS":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 168, 64
        train_task_list, test_task_list = FedCSISSeriesData(
            column=column, series=series
        )
        # train_task_list, test_task_list = FedCSISData(column=column)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maml_epoch = 30
    normal_epoch = 30

    # test maml first
    maml_result = train(
        "maml",
        test_task_list,
        maml_epoch,
        device,
        dataset_name,
        inchannel,
        out_channel,
        in_len,
        hidden_size,
        column,
        model_type,
        series,
    )
    normal_result = train(
        "normal",
        test_task_list,
        normal_epoch,
        device,
        dataset_name,
        inchannel,
        out_channel,
        in_len,
        hidden_size,
        column,
        model_type,
        series,
    )
    plot_result(dataset_name, normal_result, maml_result, column, model_type, series)
    print("Test end.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
