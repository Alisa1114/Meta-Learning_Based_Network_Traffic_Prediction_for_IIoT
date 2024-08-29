import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


def relative_rmse(labels, predictions):
    rmse = mean_squared_error(labels, predictions, squared=False)
    mean = np.mean(labels)
    if mean != 0:
        rmse /= mean
    return rmse


def relative_mae(labels, predictions):
    mae = mean_absolute_error(labels, predictions)
    mean = np.mean(labels)
    if mean != 0:
        mae /= mean
    return mae


def reverse_scale(data, mean, std):
    if std != 0:
        data = data * std + mean
    return data


def plot_predictions(dataset_name, pred, label, dataset_num):
    dir_name = dataset_name + "_test_prediction_plot"
    os.makedirs(dir_name, exist_ok=True)
    plt.figure(figsize=(100, 20))
    plt.figure()
    plt.plot(pred, color="green", label="Prediction")
    plt.plot(label, color="red", label="Label")
    plt.title("Time Series {}".format(dataset_num))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(dir_name + "/time_series_{}.png".format(dataset_num))
    plt.close()


def plot_result(dataset_name, baseline, maml, column, model_type, series):
    if dataset_name == "FedCSIS":
        dir_name = "exp/{}_{}_{}/{}".format(dataset_name, series, column, model_type)
    else:
        dir_name = "exp/{}/{}".format(dataset_name, model_type)
    os.makedirs(dir_name, exist_ok=True)
    num_dataset = len(baseline)

    total_maml_mape = 0.0
    total_maml_rmse = 0.0
    total_maml_r2 = 0.0
    total_maml_mae = 0.0
    total_baseline_mape = 0.0
    total_baseline_rmse = 0.0
    total_baseline_r2 = 0.0
    total_baseline_mae = 0.0

    for i, (baseline_result, maml_result) in enumerate(zip(baseline, maml)):
        # plt.figure(figsize=(10, 5))
        plt.figure(figsize=(7, 3.5))
        # plt.subplot(1, 4, 1)
        # plt.plot(baseline_result["mape"], color="green", label="baseline")
        # plt.plot(maml_result["mape"], color="red", label="MAML")
        # plt.title("MAPE")
        # plt.xlabel("Epoch")
        # plt.ylabel("MAPE")
        # plt.legend()
        # total_maml_mape += np.array(maml_result["mape"])
        # total_baseline_mape += np.array(baseline_result["mape"])

        plt.subplot(1, 2, 1)
        plt.plot(baseline_result["rmse"], color="green", label="Baseline")
        plt.plot(maml_result["rmse"], color="red", label="MAML")
        plt.title("RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        total_maml_rmse += np.array(maml_result["rmse"])
        total_baseline_rmse += np.array(baseline_result["rmse"])

        # plt.subplot(1, 4, 3)
        # plt.plot(baseline_result["r2"], color="green", label="baseline")
        # plt.plot(maml_result["r2"], color="red", label="MAML")
        # plt.title("R2")
        # plt.xlabel("Epoch")
        # plt.ylabel("R2")
        # plt.legend()
        # total_maml_r2 += np.array(maml_result["r2"])
        # total_baseline_r2 += np.array(baseline_result["r2"])

        plt.subplot(1, 2, 2)
        plt.plot(baseline_result["mae"], color="green", label="Baseline")
        plt.plot(maml_result["mae"], color="red", label="MAML")
        plt.title("MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        
        total_maml_mae += np.array(maml_result["mae"])
        total_baseline_mae += np.array(baseline_result["mae"])

        plt.tight_layout()
        plt.savefig(dir_name + "/time_series_{}.png".format(i + 1))
        plt.close()

    # plt.figure(figsize=(10, 5))
    plt.figure(figsize=(7, 3.5))
    # plt.subplot(1, 4, 1)
    # plt.plot(total_baseline_mape / num_dataset, color="green", label="baseline")
    # plt.plot(total_maml_mape / num_dataset, color="red", label="MAML")
    # plt.title("MAPE")
    # plt.xlabel("Epoch")
    # plt.ylabel("MAPE")
    # plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(total_baseline_rmse / num_dataset, color="green", label="Baseline")
    plt.plot(total_maml_rmse / num_dataset, color="red", label="MAML")
    plt.title("RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()

    # plt.subplot(1, 4, 3)
    # plt.plot(total_baseline_r2 / num_dataset, color="green", label="baseline")
    # plt.plot(total_maml_r2 / num_dataset, color="red", label="MAML")
    # plt.title("R2")
    # plt.xlabel("Epoch")
    # plt.ylabel("R2")
    # plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(total_baseline_mae / num_dataset, color="green", label="Baseline")
    plt.plot(total_maml_mae / num_dataset, color="red", label="MAML")
    plt.title("MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    # if dataset_name == "FedCSIS":
    #     plt.suptitle("F-{}".format(series))
    # elif dataset_name == "Guangzhou":
    #     plt.suptitle("G-dataset")
    # elif dataset_name == "Seattle":
    #     plt.suptitle("S-dataset")
    plt.tight_layout()
    plt.savefig(dir_name + "/average.png")
    plt.close()
