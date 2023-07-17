from .dcrnn import DCRNN
from .fbf_gru import FBF_GRU
from .fbf_lstm import FBF_LSTM
from .gru import GRU
from .gwn import GWNet, GraphConvNet
from .lstm import LSTM
from .mtgnn import MTGNN
from .stgcn import STGCN
from .transformer_tm import Transformer_TM
from .vae import VAE


def create_model(args):
    if args.model == 'gwn':
        return GWNet.from_args(args).to(args.device)
    elif args.model == 'lstm':
        return LSTM(args).to(args.device)
    elif args.model == 'gru':
        return GRU(args).to(args.device)
    elif args.model == 'fbf_lstm':
        return FBF_LSTM(args).to(args.device)
    elif args.model == 'fbf_gru':
        return FBF_GRU(args).to(args.device)
    elif args.model == 'transformer':
        return Transformer_TM(args).to(args.device)
    elif args.model == 'stgcn':
        return STGCN(args).to(args.device)
    elif args.model == 'dcrnn':
        return DCRNN(args).to(args.device)
    elif args.model == 'mtgnn':
        return MTGNN(args).to(args.device)
    elif args.model == 'vae':
        return VAE(args=args)
    else:
        raise NotImplementedError
