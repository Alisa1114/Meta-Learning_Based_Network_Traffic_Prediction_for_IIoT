import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from lstm import SimpleLSTM
from attention import SelfAttention


class CNN1(nn.Module):
    def __init__(self, in_channels=1, out_channel=1, input_len=168, hidden_size=64):
        super(CNN1, self).__init__()
        seq_len = (input_len // 8) * hidden_size * 4
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, 5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size, hidden_size * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, 3, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self, in_channels=1, out_channel=1, input_len=288, hidden_size=8):
        super(CNN2, self).__init__()
        seq_len = (input_len - 4 - 4 - 2 - 2) * hidden_size * 8
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, 5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size * 2, 5),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, 3),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size * 4, hidden_size * 8, 3),
            nn.BatchNorm1d(hidden_size * 8),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


#  CNN-LSTM
class LSTM(nn.Module):
    def __init__(self, in_channels=1, out_channel=1, input_len=168, hidden_size=64):
        super(LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, 5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size, hidden_size * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, 3, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = SimpleLSTM(
            hidden_size * 4, hidden_size * 4, num_layers=2, dropout=0.4
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, out_channel),
        )

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x.permute(0, 2, 1))
        x = self.fc(x[:, -1, :])
        return x


# class LSTM(nn.Module):
#     def __init__(self, in_channels=1, out_channel=1, input_len=168, hidden_size=64):
#         super(LSTM, self).__init__()
#         self.lstm = SimpleLSTM(in_channels, hidden_size, num_layers=2, dropout=0.4)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 32),
#             nn.Dropout(0.4),
#             nn.ReLU(inplace=True),
#             nn.Linear(32, out_channel),
#         )

#     def forward(self, x):
#         x, _ = self.lstm(x.permute(0, 2, 1))
#         x = self.fc(x[:, -1, :])
#         return x


class Transformer(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_size, num_heads=1, num_layers=1):
        super(Transformer, self).__init__()
        self.in_fc = nn.Linear(in_channel, hidden_size)
        self.layers = [SelfAttention(hidden_size, num_heads) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_channel),
        )

    def forward(self, x):
        x = self.in_fc(x.permute(0, 2, 1))
        for attn in self.layers:
            x = attn(x)
        x = self.fc(x[:, -1, :])
        return x


def get_model(dataset_name, in_channel, out_channel, in_len, hidden_size, model_type):
    if model_type == "attn":
        model = Transformer(in_channel, out_channel, hidden_size)
    elif model_type == "lstm":
        model = LSTM(in_channel, out_channel, in_len, hidden_size)
    else:
        if dataset_name == "FedCSIS":
            model = CNN1(in_channel, out_channel, in_len, hidden_size)
        else:
            model = CNN2(in_channel, out_channel, in_len, hidden_size)
    return model


if __name__ == "__main__":
    dataset_name = "Guangzhou"
    in_len = 144
    hidden_size = 8
    model_type = "attn"
    model = get_model(dataset_name,
              1,
              1,
              in_len,
              hidden_size,
              model_type)
    print(model)
