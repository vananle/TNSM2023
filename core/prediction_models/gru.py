import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.num_flow = args.num_mon_flow
        self.in_dim = self.num_flow

        hidden = args.hidden
        num_layer = args.layers
        dropout = args.dropout
        self.output_len = args.output_len

        self.gru = nn.GRU(
            self.in_dim, hidden, num_layer, batch_first=True, dropout=dropout)
        print(self.gru)
        self.fc = nn.Linear(hidden, self.num_flow * self.output_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        x, _ = self.gru(x)  # BS,T,h
        x = x[:, -1]
        x = self.fc(x)  # BS, out_len*f
        x = torch.reshape(x, shape=(x.size(0), self.output_len, self.num_flow))
        return x
