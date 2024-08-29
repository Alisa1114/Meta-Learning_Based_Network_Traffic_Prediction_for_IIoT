import torch
from model import get_model
from torchinfo import summary
import os


def main(
    epochs, num_tasks, dataset_name, learning_rate, meta_lr, column, model_type, series
):
    os.makedirs("model_size", exist_ok=True)
    file_name = "model_size/{}_{}.txt".format(dataset_name, model_type)
    
    if dataset_name == "Guangzhou":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
    elif dataset_name == "Seattle":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
    elif dataset_name == "FedCSIS":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 168, 64

    model = get_model(
        dataset_name, inchannel, out_channel, in_len, hidden_size, model_type
    )
    input_size = (128, 1, in_len)
    model_info = summary(model, input_size=input_size)
    
    with open(file_name, "w") as f:
        f.write(str(model_info))
    
    

if __name__ == "__main__":
    epochs = 30
    num_tasks = 128  # 128 for FedCSIS, 64
    dataset_name = "Seattle"  # FedCSIS, Guangzhou, Seattle
    learning_rate = 5e-4
    meta_lr = 1e-3
    column = "Volume"  # Mean, Volume
    # series_list = [
    #     "memoory_free",
    #     "cpu_usage",
    #     "memoryallocatedbyproc",
    #     "in_traffic",
    #     "out_traffic",
    # ]
    series = "out_traffic"
    model_type = "attn"  # cnn, lstm, attn
    main(
        epochs,
        num_tasks,
        dataset_name,
        learning_rate,
        meta_lr,
        column,
        model_type,
        series,
    )
