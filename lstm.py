import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 遗忘门参数
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        # 输入门参数
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        # 细胞状态参数
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)
        self.bc = nn.Parameter(torch.zeros(hidden_size))

        # 输出门参数
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        # 初始化隱藏狀態和細胞狀態
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden

        # 將輸入 x 在時間維度上展開
        x = x.permute(
            1, 0, 2
        )  # 將 (batch_size, seq_len, input_size) 轉換為 (seq_len, batch_size, input_size)

        # 將所有時間步的操作一次性進行
        combined = torch.cat((x, h_t.unsqueeze(0).expand(seq_len, -1, -1)), dim=2)

        f_t = torch.sigmoid(
            self.Wf(combined)
            + self.Uf(h_t).unsqueeze(0).expand(seq_len, -1, -1)
            + self.bf
        )
        i_t = torch.sigmoid(
            self.Wi(combined)
            + self.Ui(h_t).unsqueeze(0).expand(seq_len, -1, -1)
            + self.bi
        )
        o_t = torch.sigmoid(
            self.Wo(combined)
            + self.Uo(h_t).unsqueeze(0).expand(seq_len, -1, -1)
            + self.bo
        )
        c_hat_t = torch.tanh(
            self.Wc(combined)
            + self.Uc(h_t).unsqueeze(0).expand(seq_len, -1, -1)
            + self.bc
        )

        # 運用 PyTorch 的向量化操作，一次性計算所有時間步的細胞狀態和隱藏狀態
        c_t = torch.sum(f_t * c_t.unsqueeze(0).expand_as(f_t) + i_t * c_hat_t, dim=0)
        h_t = o_t * torch.tanh(c_t)

        # 將隱藏狀態再轉置回 (batch_size, seq_len, hidden_size)
        return h_t.permute(1, 0, 2), (h_t[-1], c_t)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(SimpleLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = []
        for _ in range(num_layers):
            self.lstm_layers.append(LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

    def forward(self, x):
        hidden = None
        for i in range(self.num_layers):
            x, hidden = self.lstm_layers[i](x, hidden)
            if (i + 1) != self.num_layers:
                x = self.dropout(x)
        return x, hidden
