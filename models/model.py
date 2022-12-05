import torch 
from torch import nn
from .conv_tpp.model import ConvTPP

class TPP(nn.Module):
    def __init__(self, config):
        super(TPP, self).__init__()
        num_types = config['num_types']
        self.num_types = num_types
        model_name = config['model']
        time_ratio = config['time_ratio'] if 'time_ratio' in config else 1
        self.time_ratio = time_ratio
        if model_name.lower() == 'conv-tpp':
            self.model = ConvTPP(config)
        else:
            raise NotImplementedError(f'{model_name} is not implemented.')
        self.register_buffer('device_indicator', torch.empty(0))

    def compute_loss(self, type_seq, time_seq):
        type_seq, time_seq = processSeq(type_seq, time_seq, self.time_ratio)
        loss, type_loss, dt_loss = self.model.compute_loss(type_seq, time_seq)
        return loss, type_loss, dt_loss

    def predict(self, type_seq, time_seq): 
        return self.model.predict(type_seq, time_seq)

    def plot(self, dir_name='default', width=1):
        self.model.plot(dir_name, width)

def processSeq(type_seq, time_seq, time_ratio=1): # TODO: no longer predict or calculate loss for the first event
    time_seq = time_seq - time_seq[:, 0].view(-1, 1)
    time_seq = time_seq / time_ratio
    mask = type_seq.eq(0)
    time_seq.masked_fill_(mask, 0)
    return type_seq, time_seq
