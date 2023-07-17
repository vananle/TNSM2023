import torch
import torch.nn as nn


class FBF_LSTM(nn.Module):
    def __init__(self, args):
        super(FBF_LSTM, self).__init__()

        self.in_dim = 1
        self.num_flow = args.num_mon_flow
        self.hidden = args.hidden
        num_layer = args.layers
        dropout = args.dropout
        self.output_len = args.output_len
        self.lstm = nn.LSTM(
            self.in_dim, self.hidden, num_layer, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden, self.output_len)

    def forward(self, x):
        # BS, seq_len, f = x.size()
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (-1, x.size(-1), 1))
        x, _ = self.lstm(x)  # BS,T,h
        x = x[:, -1]
        x = self.fc(x)  # BS, out_len*f
        x = torch.reshape(x, (-1, self.num_flow, self.output_len))
        x = torch.permute(x, (0, 2, 1))
        return x
