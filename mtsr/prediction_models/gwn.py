import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, num_flows, batch_size, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, output_len=12,
                 residual_channels=32, dilation_channels=32, cat_feat_gc=False,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, stride=2,
                 apt_size=10, verbose=0):
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj

        self.batch_size = batch_size
        self.num_flows = num_flows

        self.verbose = verbose

        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d(in_channels=in_dim - 1,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_flows, apt_size), torch.randn(apt_size, num_flows)
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, 1) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, 1) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList(
            [GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
             for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, kernel_size, dilation=D))
                D *= stride
                receptive_field += additional_scope
                additional_scope *= stride
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, output_len, (1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, args):

        dropout = args.dropout
        do_graph_conv = True
        addaptadj = True
        in_dim = args.in_dim
        apt_size = 10
        output_len = args.output_len
        hidden = args.hidden
        kernel_size = args.kernel_size
        stride = args.stride
        blocks = args.blocks
        layers = args.layers
        cat_feat_gc = True if in_dim > 1 else False
        verbose = args.verbose
        device = args.device
        num_node = args.num_node
        num_flow = args.num_mon_flow
        batch_size = args.train_batch_size

        defaults = dict(dropout=dropout, supports=None,
                        do_graph_conv=do_graph_conv, addaptadj=addaptadj, aptinit=None,
                        in_dim=in_dim, apt_size=apt_size, output_len=output_len,
                        residual_channels=hidden, dilation_channels=hidden,
                        stride=stride, kernel_size=kernel_size,
                        blocks=blocks, layers=layers,
                        skip_channels=hidden * 8, end_channels=hidden * 16,
                        cat_feat_gc=cat_feat_gc, verbose=verbose, device=device,
                        num_nodes=num_node, num_flows=num_flow, batch_size=batch_size)
        # defaults.update(**kwargs)
        model = cls(**defaults)
        return model

    def load_checkpoint(self, state_dict):
        """It is assumed that ckpt was trained to predict a subset of timesteps."""
        bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']  # only weights that depend on seq_length
        b, w = state_dict.pop(bk), state_dict.pop(wk)
        self.load_state_dict(state_dict, strict=False)
        cur_state_dict = self.state_dict()
        cur_state_dict[bk][:b.shape[0]] = b
        cur_state_dict[wk][:w.shape[0]] = w
        self.load_state_dict(cur_state_dict)

    def forward(self, x):
        # input x (b, seq_x, n, features)
        x = x.transpose(1, 3)

        if self.verbose:
            print('------------------- GWN model ----------------------')
            print('Input shape: ', x.shape)
        # x (bs, features, n_nodes, n_timesteps)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        # first linear layer
        if self.cat_feat_gc and x.size(1) > 1:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x)

        if self.verbose:
            print('After first linear: ', x.shape)

        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:  # equation (6) and (7)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # the learnable adj matrix
            adjacency_matrices = self.fixed_supports + [adp]
            if self.verbose:
                for adj in adjacency_matrices:
                    print('adj shape', adj.shape)

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            if self.verbose:
                print('Layer: ', i, self.blocks, self.layers)
                print('   Input layer: ', x.shape)

            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))

            batch_size = residual.size(0)
            gate_in = torch.permute(residual, (0, 2, 1, 3))
            if self.verbose: print('gate_in: ', gate_in.size())
            gate_in = torch.reshape(gate_in, shape=(batch_size * self.num_flows, self.residual_channels, -1))
            if self.verbose: print('--> gate_in: ', gate_in.size())
            gate = torch.sigmoid(self.gate_convs[i](gate_in))
            gate = torch.reshape(gate, shape=(batch_size, self.num_flows, self.dilation_channels, -1))
            gate = torch.permute(gate, (0, 2, 1, 3))

            x = filter * gate

            if self.verbose:
                print('   filter shape: ', filter.shape)
                print('   gate shape: ', gate.shape)
                print('   Gated tcn output: ', x.shape)

            # parametrized skip connection

            skip_in = torch.permute(x, (0, 2, 1, 3))
            skip_in = torch.reshape(skip_in, shape=(batch_size * self.num_flows, self.dilation_channels, -1))
            s = self.skip_convs[i](skip_in)  # what are we skipping??
            s = torch.reshape(s, shape=(batch_size, self.num_flows, self.skip_channels, -1))
            s = torch.permute(s, (0, 2, 1, 3))

            try:  # if i > 0 this works
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.verbose:
                print('   Skip shape: ', skip.shape)

            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)

            if self.verbose:
                print('   Graph output: ', x.shape)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # downsample to (bs, seq_length, nodes, 1)
        if self.verbose:
            print('\nSkip shape: ', skip.shape)
            print('Output shape: ', x.shape)
            print('------------------------------------------------')

        return x.squeeze(dim=-1)  # (bs, seq_y, n)
