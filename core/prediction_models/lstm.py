import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.num_flow = args.num_mon_flow
        self.in_dim = self.num_flow

        hidden = args.hidden
        num_layer = args.layers
        dropout = args.dropout
        self.output_len = args.output_len
        self.lstm = nn.LSTM(
            self.in_dim, hidden, num_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, self.num_flow * self.output_len)

    def forward(self, x):
        # BS, seq_len, f = x.size()
        x, _ = self.lstm(x)  # BS,T,h
        x = x[:, -1]
        x = self.fc(x)  # BS, out_len*f
        x = torch.reshape(x, shape=(x.size(0), self.output_len, self.num_flow))
        return x
