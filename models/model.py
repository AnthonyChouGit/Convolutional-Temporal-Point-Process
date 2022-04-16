import torch 
from torch import nn
from .nhp.model import NHP
from .conv_tpp.model import ConvTPP
from .hp.model import HP
from .rmtpp.model import RMTPP
from .log_norm_mix.model import LogNormMixTPP

class TPP(nn.Module):
    def __init__(self, config):
        super(TPP, self).__init__()
        num_types = config['num_types']
        # embed_dim = config['embed_dim']
        # self.embed = nn.Embedding(num_types+2, embed_dim, padding_idx=0)
        self.num_types = num_types
        model_name = config['model']
        if model_name.lower() == 'nhp':
            self.model = NHP(config)
        elif model_name.lower() == 'conv-tpp':
            self.model = ConvTPP(config)
        elif model_name.lower() == 'hp':
            self.model = HP(config)
        elif model_name.lower() == 'rmtpp':
            self.model = RMTPP(config)
        elif model_name.lower() == 'log-norm-mix':
            self.model = LogNormMixTPP(config)
        else:
            raise NotImplementedError(f'{model_name} is not implemented.')
        self.register_buffer('device_indicator', torch.empty(0))

    def compute_loss(self, type_seq, time_seq):
        type_seq, time_seq = processSeq(type_seq, time_seq)
        loss = self.model.compute_loss(type_seq, time_seq)
        return loss


    def predict(self, type_seq, time_seq): 
        return self.model.predict(type_seq, time_seq)

def processSeq(type_seq, time_seq): # TODO: no longer predict or calculate loss for the first event
    time_seq = time_seq - time_seq[:, 0].view(-1, 1)
    mask = type_seq.eq(0)
    time_seq.masked_fill_(mask, 0)
    return type_seq, time_seq
