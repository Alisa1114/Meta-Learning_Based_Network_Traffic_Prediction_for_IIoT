import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import learn2learn as l2l
import random
from model import get_model
from dataset import meta_dataset
from dataset import GuangzhouData, SeattleData, FedCSISData
from dataset import GuangzhouDataShort, SeattleDataShort, FedCSISSeriesData


def main(
    epochs,
    num_tasks,
    dataset_name,
    learning_rate,
    meta_lr,
    column,
    model_type,
    series,
    data_len,
):
    print("Train {}".format(dataset_name))
    if dataset_name == "Guangzhou":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
        # train_task_list, test_task_list = GuangzhouData(
        #     x_length=in_len, y_length=out_channel
        # )
        train_task_list, test_task_list = GuangzhouDataShort(
            x_length=in_len, query_rate=0.2, total_length=data_len
        )

    elif dataset_name == "Seattle":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 144, 8
        # train_task_list, test_task_list = SeattleData(
        #     x_length=in_len, y_length=out_channel
        # )
        train_task_list, test_task_list = SeattleDataShort(
            x_length=in_len, query_rate=0.2, total_length=data_len
        )

    elif dataset_name == "FedCSIS":
        inchannel, out_channel, in_len, hidden_size = 1, 1, 168, 64
        # train_task_list, test_task_list = FedCSISData(column=column)
        train_task_list, test_task_list = FedCSISSeriesData(
            column=column, series=series
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = num_tasks if len(train_task_list) >= num_tasks else len(train_task_list)
    model = get_model(
        dataset_name, inchannel, out_channel, in_len, hidden_size, model_type
    )
    model = model.to(device)
    maml = l2l.algorithms.MAML(model, lr=meta_lr)
    optimizer = optim.Adam(maml.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        selected_tasks = random.sample(train_task_list, k=num_tasks)

        for i, task in enumerate(selected_tasks):
            supp_dataset = task["train"]
            query_dataset = task["test"]
            mean, std = task["mean"], task["std"]
            supp_loader = DataLoader(
                supp_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            qry_loader = DataLoader(
                query_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            step_loss = 0.0
            learner = maml.clone()

            for X, y, _ in supp_loader:
                X = X.to(device)
                y = y.to(device)
                output = learner(X)
                loss = criterion(output, y)
                learner.adapt(loss)

            for X, y, _ in qry_loader:
                X = X.to(device)
                y = y.to(device)
                output = learner(X)
                adapt_loss = criterion(output, y)
                step_loss += adapt_loss

            step_loss = step_loss / len(qry_loader)
            step_loss.backward()
            print(
                "Epoch: {} Task: {} Loss: {}".format(epoch + 1, i + 1, step_loss.item())
            )

        for p in maml.parameters():
            p.grad.data.mul_(1.0 / num_tasks)

        optimizer.step()

    if dataset_name == "FedCSIS":
        weight_name = "weights/{}-{}-{}-{}.pth".format(
            dataset_name, series, column, model_type
        )
    else:
        weight_name = "weights/{}-{}-{}.pth".format(dataset_name, model_type, data_len)
    torch.save(maml.module.state_dict(), weight_name)


if __name__ == "__main__":
    epochs = 30
    num_tasks = 64  # 128 for FedCSIS, 64
    dataset_name = "Guangzhou"  # FedCSIS, Guangzhou, Seattle
    learning_rate = 5e-4
    meta_lr = 1e-3
    column = "Volume"  # Mean, Volume
    # series_list = [
    #     "cpu_usage",
    #     "memoory_free",
    #     "memoryallocatedbyproc",
    #     "in_traffic",
    #     "out_traffic",
    # ]
    series = "out_traffic"
    model_type = "attn"  # cnn, lstm, attn
    data_len = 3000
    main(
        epochs,
        num_tasks,
        dataset_name,
        learning_rate,
        meta_lr,
        column,
        model_type,
        series,
        data_len,
    )
