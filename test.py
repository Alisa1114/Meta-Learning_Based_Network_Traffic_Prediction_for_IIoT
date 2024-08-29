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
import os
from model import get_model
from dataset import meta_dataset
from dataset import GuangzhouData, SeattleData, FedCSISData
from dataset import GuangzhouDataShort, SeattleDataShort, FedCSISSeriesData
from utils import plot_predictions, reverse_scale


@torch.no_grad()
def test(
    test_dataset,
    model,
    device,
    mean,
    std,
    dataset_name,
    dataset_num,
    plot=True,
):
    data_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )
    predictions = []
    labels = []

    for X, _, y in data_loader:
        input_data = X.to(device)
        output = model(input_data)
        output = output.cpu().numpy()
        y = y.numpy()
        predictions.append(output)
        labels.append(y)

    predictions = np.concatenate(predictions, axis=0)
    predictions = reverse_scale(predictions, mean, std)
    labels = np.concatenate(labels, axis=0)

    # replace zeros in labels with 0.0001, so mape won't be very high
    labels_process = np.copy(labels)
    labels_process[labels_process == 0] = 1e-4

    rmse = mean_squared_error(labels, predictions, squared=False)
    mae = mean_absolute_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels_process, predictions)
    r2 = r2_score(labels, predictions)

    if plot:
        plot_predictions(dataset_name, predictions, labels, dataset_num)

    return mape, rmse, r2, mae


def main():
    training = "normal"
    model_type = "attn"
    dataset_name = "Guangzhou"
    data_len = 3000
    column = "Volume"
    series = "out_traffic"
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
        train_task_list, test_task_list = FedCSISSeriesData(
            column=column, series=series
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10

    if dataset_name == "FedCSIS":
        os.makedirs(
            "record/{}/{}_{}_{}/{}epoch".format(
                training, dataset_name, series, column, epochs
            ),
            exist_ok=True,
        )
        file_name = "record/{}/{}_{}_{}/{}epoch/{}.txt".format(
            training, dataset_name, series, column, epochs, model_type
        )
    else:
        os.makedirs(
            "record/{}/{}/{}/{}epoch".format(training, dataset_name, data_len, epochs),
            exist_ok=True,
        )
        file_name = "record/{}/{}/{}/{}epoch/{}.txt".format(
            training, dataset_name, data_len, epochs, model_type
        )
    test_record = open(file_name, mode="w")

    total_rmse, total_r2, total_mape, total_mae = 0.0, 0.0, 0.0, 0.0

    for i, task in enumerate(test_task_list):
        train_dataset = task["train"]
        test_dataset = task["test"]
        mean, std = task["mean"], task["std"]
        data_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

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
                    torch.load(
                        "weights/{}-{}-{}.pth".format(
                            dataset_name, model_type, data_len
                        )
                    )
                )
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
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
            plot=False,
        )
        result = "Dataset: {} MAPE: {:.4f} RMSE: {:.4f} R2: {:.4f} MAE: {:.4f}".format(
            i + 1, mape, rmse, r2, mae
        )
        print(result)
        test_record.write(result + "\n")

        total_mape += mape
        total_r2 += r2
        total_rmse += rmse
        total_mae += mae

    result = "Final Result\nAvg MAPE: {} Avg RMSE: {} Avg R2: {} Avg MAE: {}".format(
        total_mape / len(test_task_list),
        total_rmse / len(test_task_list),
        total_r2 / len(test_task_list),
        total_mae / len(test_task_list),
    )
    print(result)
    test_record.write(result + "\n")
    test_record.close()


if __name__ == "__main__":
    main()
