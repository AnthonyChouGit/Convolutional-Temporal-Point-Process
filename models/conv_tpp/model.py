import torch
from torch import nn, no_grad
from .modules.conv import LocalConv
from .modules.lognorm import LogNormMix
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os

class ConvTPP(nn.Module):
    def __init__(self, config):
        """
        Required arguments:
            num_types, embed_dim, hidden_dim, num_channel, horizon, num_component,
        """
        super(ConvTPP, self).__init__()
        num_types = config['num_types']
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_channel = config['num_channel']
        horizon = config['horizon']
        num_component = config['num_component']
        plot_samples = config['plot_samples'] if 'plot_samples' in config else 10000
        omega = config['omega']
        self.horizon = horizon
        self.num_channel = num_channel
        self.plot_samples = plot_samples
        self.embed = nn.Embedding(num_types+1, embed_dim, padding_idx=0)
        siren_dim = config['siren_dim'] if 'siren_dim' in config else 32
        siren_layers = config['siren_layers'] if 'siren_layers' in config else 3
        self.conv = LocalConv(d_model=embed_dim, siren_hid=siren_dim, 
                            siren_hid_num=siren_layers, num_channel=num_channel, 
                            horizon=horizon, omega=omega)
        self.rnn = nn.GRU(input_size=embed_dim+1, hidden_size=hidden_dim, 
                    batch_first=True)
        self.time_stack = nn.Linear(hidden_dim, 3*num_component)
        self.mark_stack = nn.Linear(hidden_dim, num_types)
        self.num_component = num_component
        self.register_buffer('device_indicator', torch.empty(0))

    def encode(self, type_seq: torch.tensor, time_seq: torch.tensor)->torch.tensor:
        device = self.device_indicator.device
        embed_seq = self.embed(type_seq)
        batch_size, seq_len = type_seq.size()
        mask = type_seq.eq(0) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
        dtimes.masked_fill_(mask[:, 1:], 0)
        dtimes.clamp_(1e-10)
        embed_seq = self.conv(embed_seq, time_seq, mask) # 1-seq_len
        temporal = torch.cat([torch.ones(batch_size, 1, device=device)*1e-10, dtimes], dim=1) #TODO: .log() 1-seq_len
        embed_seq = torch.cat([embed_seq, temporal.unsqueeze(-1)], dim=-1) # 1-seq_len
        self.rnn.flatten_parameters()
        all_encs = self.rnn(embed_seq)[0] # (batch_size, seq_len, hidden_dim) 1-seq_len
        return all_encs

    def decode(self, all_encs: torch.tensor) -> tuple:
        dist_params = self.time_stack(all_encs)
        locs = dist_params[...,:self.num_component]
        log_scales = dist_params[...,self.num_component:(2*self.num_component)]
        log_weights = dist_params[...,(2*self.num_component):]
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormMix(locs=locs, log_scales=log_scales, log_weights=log_weights)
        type_prob = torch.log_softmax(self.mark_stack(all_encs), dim=-1)
        type_dist = Categorical(logits=type_prob)
        return type_dist, inter_time_dist

    def compute_loss(self, type_seq: torch.tensor, time_seq: torch.tensor) -> tuple: # 1-seq_len
        device = self.device_indicator.device
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device) # 1-seq_len
        all_encs = self.encode(type_seq[:, :-1], time_seq[:, :-1]) # 1-seq_len-1

        mask = type_seq.eq(0) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
        dtimes.masked_fill_(mask[:, 1:], 0)
        dtimes.clamp_(1e-10)

        type_dist, inter_time_dist = self.decode(all_encs) # 2-seq_len

        time_log_probs = inter_time_dist.log_prob(dtimes) # 2-seq_len
        time_log_probs.masked_fill_(mask[:, 1:], 0)
        temp_mark = (type_seq.masked_fill(mask, 1) - 1)[:, 1:] # 2-seq_len
        type_log_probs = type_dist.log_prob(temp_mark) # 2-seq_len
        type_log_probs.masked_fill_(mask[:, 1:], 0)
        time_loss = -time_log_probs.sum()
        type_loss = -type_log_probs.sum()
        loss = time_loss + type_loss
        return loss, type_loss, time_loss

    def predict(self, type_seq, time_seq):
        raise NotImplementedError()

    def plot(self, dir_name='default', width=1):
        self.eval()
        with torch.no_grad():
            device = self.device_indicator.device
            num_layers = len(self.horizon)
            if not os.path.exists(f'figs/{dir_name}'):
                os.mkdir(f'figs/{dir_name}')
            for i in range(num_layers):
                if isinstance(self.horizon[i], list):
                    max_dt = torch.max(self.horizon[i]).item()
                else:
                    max_dt = self.horizon[i]
                dt_vals = torch.linspace(0, max_dt, self.plot_samples+1).to(device)
                kern_vals = self.conv.layers[i].kernel_forward(dt_vals)
                dt_vals = dt_vals.to('cpu').numpy()
                kern_vals = kern_vals.to('cpu').numpy()
                plt.rcParams['figure.figsize'] = (6, 6)
                # plt.locator_params(axis='x', nbins=5)
                for j in range(kern_vals.shape[1]):
                    plt.plot(dt_vals, kern_vals[:, j], linewidth=width)
                # plt.xlabel('Δt', fontsize=32)
                # plt.ylabel('ψ(Δt)', fontsize=32)
                # plt.title(f'layer {i}', fontsize=25)

                plt.yticks([-0.457, -0.456, -0.455, -0.454],fontsize=32)
                plt.xticks([0, 0.2, 0.4, 0.6, 0.8], labels=[0 ,2,4,6,8], fontsize=32)
                plt.xlabel('$τ$ (×$10^{-1}$)', fontsize=32)
                plt.ylabel('ψ(τ)', fontsize=32)


                # plt.xticks(fontsize=32)
                # plt.yticks(fontsize=32)
                plt.savefig(f'figs/{dir_name}/layer {i}.jpg', bbox_inches='tight')
                plt.clf()
        self.train()
