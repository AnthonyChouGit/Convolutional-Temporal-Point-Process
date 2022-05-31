from torch import nn
import torch
from torch.nn.utils import weight_norm
from .utils.mask import get_subsequent_mask

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)

class SirenNet(nn.Module):
    def __init__(self, d_hidden, hid_num, num_channel):
        super(SirenNet, self).__init__()
        self.in_mapping = nn.Sequential(
            weight_norm(nn.Linear(1, d_hidden)),
            Sine(),
            nn.Dropout(0.1)
        )

        self.hid_stack = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Linear(d_hidden, d_hidden)),
                Sine(),
                nn.Dropout(0.1)
            )
            for _ in range(hid_num-1)
        ])

        self.out_mapping = weight_norm(nn.Linear(d_hidden, num_channel))

    def forward(self, tau):
        """
            tau: (*, 1)
            output: 
                (*, num_channel)
        """
        x = self.in_mapping(tau)
        if len(self.hid_stack)>0:
            for layer in self.hid_stack:
                x = layer(x)
        x = self.out_mapping(x)
        return x

class LocalConvLayer(nn.Module):
    def __init__(self, d_model, d_hid, hid_num, num_channel, horizon):
        super(LocalConvLayer, self).__init__()
        self.siren = SirenNet(d_hid, hid_num, num_channel)
        self.horizon = horizon
        self.out_mapping = nn.Linear(num_channel*d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.num_channel = num_channel

    def kernel_forward(self, dt):
        """
            dt: (n)
        """
        if isinstance(self.horizon, list):
            # horizon:(num_channel,)
            # horizon_mask: (n, num_channel)
            horizon_mask = (dt[:, None] > torch.tensor(self.horizon, device=dt.device)[None, :])
        else:
            # horizon: 1
            # horizon_mask: (n, 1)
            horizon_mask = (dt > self.horizon).unsqueeze(-1)
        kernel = self.siren(dt.unsqueeze(-1)) # (n, num_channel)
        kernel.masked_fill_(horizon_mask, 0)
        return kernel

    def forward(self, embed_seq, time_seq, mask):
        """
            embed_seq: (batch_size, seq_len, d_model)
            time_seq: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        """
        batch_size, seq_len, d_model = embed_seq.size()
        embed_seq = embed_seq.masked_fill(mask[:, :, None], 0)
        # rel_tau: (batch_size, seq_len, seq_len)
        rel_tau = time_seq[:, :, None] - time_seq[:, None, :]
        
        sub_mask = get_subsequent_mask(seq_len).to(rel_tau.device)
        rel_tau.masked_fill_(mask[:, :, None], -1)
        rel_tau.masked_fill_(mask[:, None, :], -1)
        rel_tau.masked_fill_(sub_mask, -1)
        # horizon_mask: (batch_size, seq_len, seq_len) / (batch_size, seq_len, seq_len, num_channel)   
        if isinstance(self.horizon, list):# (batch_size, seq_len, seq_len, 1)    (batch_size, seq_len, seq_len, num_channel)
            assert self.num_channel == len(self.horizon)
            horizon_mask = torch.logical_or(rel_tau[:, :, :, None]<0, rel_tau[:, :, :, None]>torch.tensor(self.horizon, device=rel_tau.device))
        else:   
            # (batch_size, seq_len, seq_len)
            horizon_mask = torch.logical_or(rel_tau<0, rel_tau>self.horizon)
        # assert torch.all(horizon_mask.masked_select(sub_mask))

        kernel = self.siren(rel_tau.unsqueeze(-1)) # (batch_size, seq_len, seq_len, num_channel)
        kernel.masked_fill_(mask[:, :, None, None], 0)
        kernel.masked_fill_(mask[:, None, :, None], 0)
        if isinstance(self.horizon, list):
            kernel.masked_fill_(horizon_mask, 0)
        else:
            kernel.masked_fill_(horizon_mask[:, :, :, None], 0)
        kernel = kernel.permute(0, 3, 1, 2)
        # kernel = kernel.transpose(-1, 1) # (batch_size, num_channel, seq_len, seq_len)
        
        # print(kernel.masked_select(sub_mask))
        # assert torch.all(kernel.masked_select(sub_mask)==0)

        # x: (batch_size, num_channel, seq_len, d_model)
        x = torch.matmul(kernel, embed_seq[:, None, :, :])
        x = x.transpose(1, 2) # (batch_size, seq_len, num_channel, d_model)
        x = x.reshape(batch_size, seq_len, -1)
        x = self.out_mapping(x)
        return self.norm(x + embed_seq)

class LocalConv(nn.Module):
    def __init__(self, d_model, siren_hid, siren_hid_num, num_channel, horizon):
        super(LocalConv, self).__init__()
        num_layers = len(horizon)
        self.layers = nn.ModuleList([
            LocalConvLayer(d_model, siren_hid, siren_hid_num, num_channel, horizon[i])
            for i in range(num_layers)
        ])

    def forward(self, embed_seq, time_seq, mask):
        x = embed_seq
        for layer in self.layers:
            x = layer(x, time_seq, mask)
        return x
