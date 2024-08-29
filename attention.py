import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, drop_out=0.3):
        super(SelfAttention, self).__init__()
        self.hidden_size = input_dim // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(self.hidden_size, input_dim)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim**0.5)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.drop_out(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, dim)
        context = self.drop_out(context)
        output = self.out(context)
        return output
