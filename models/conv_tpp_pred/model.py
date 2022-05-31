import torch
from torch import nn, no_grad
from .modules.conv import LocalConv
from .modules.lognorm import LogNormMix
from torch.distributions import Categorical
import matplotlib.pyplot as plt 

class ConvTPP_Pred(nn.Module):
    def __init__(self, config):
        """
        Required arguments:
            num_types, embed_dim, hidden_dim, num_channel, horizon, num_component,
        """
        super(ConvTPP_Pred, self).__init__()
        num_types = config['num_types']
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_channel = config['num_channel']
        horizon = config['horizon']
        num_component = config['num_component']
        plot_samples = config['plot_samples'] if 'plot_samples' in config else 10000
        self.horizon = horizon
        self.num_channel = num_channel
        self.plot_samples = plot_samples
        self.embed = nn.Embedding(num_types+1, embed_dim, padding_idx=0)
        siren_dim = config['siren_dim'] if 'siren_dim' in config else 32
        siren_layers = config['siren_layers'] if 'siren_layers' in config else 3
        self.gamma = config['gamma'] if 'gamma' in config else 1
        self.conv = LocalConv(d_model=embed_dim, siren_hid=siren_dim, 
                            siren_hid_num=siren_layers, num_channel=num_channel, 
                            horizon=horizon)
        self.rnn = nn.GRU(input_size=embed_dim+1, hidden_size=hidden_dim, 
                    batch_first=True)
        self.time_stack = nn.Linear(hidden_dim, 1)
        self.mark_stack = nn.Linear(hidden_dim, num_types)
        self.num_component = num_component
        self.register_buffer('device_indicator', torch.empty(0))

    def encode(self, type_seq, time_seq):
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

    # TODO: Change code of decode, compute_loss, predict
    def decode(self, all_encs):
        type_logsoftmax = torch.log_softmax(self.mark_stack(all_encs), dim=-1)
        dtime = torch.exp(self.time_stack(all_encs)) # Need clamp?
        return type_logsoftmax, dtime

    def compute_loss(self, type_seq, time_seq):
        device = self.device_indicator.device
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device) # 1-seq_len
        all_encs = self.encode(type_seq[:, :-1], time_seq[:, :-1]) # 1-seq_len-1

        mask = type_seq.eq(0) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
        dtimes.masked_fill_(mask[:, 1:], 0)
        dtimes.clamp_(1e-10)
        # type_logsoftmax: (batch_size, seq_len, num_types), dtime: (batch_size, seq_len, 1)
        type_logsoftmax, dtimes_pred = self.decode(all_encs)
        dtimes_pred = dtimes_pred.squeeze(-1)
        temp_mark = (type_seq.masked_fill(mask, 1) - 1)[:, 1:]
        type_dist = Categorical(logits=type_logsoftmax)
        type_ce = type_dist.log_prob(temp_mark)
        type_ce.masked_fill_(mask[:, 1:], 0)
        time_error = (dtimes_pred-dtimes)**2
        time_error.masked_fill_(mask[:, 1:], 0)
        type_loss = -type_ce.sum()
        time_loss = time_error.sum()
        loss = type_loss + self.gamma * time_loss
        return loss, type_loss, time_loss

    def predict(self, type_seq, time_seq):
        device = self.device_indicator.device
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device) # 1-seq_len
        mask = type_seq.eq(0)
        all_encs = self.encode(type_seq, time_seq) # 1-seq_len
        type_logsoftmax, dtimes_pred = self.decode(all_encs)
        type_pred = type_logsoftmax.max(dim=-1)[0]
        dtimes_pred = dtimes_pred.squeeze(-1)
        type_pred.masked_fill_(mask[:, 1:], 0)
        dtimes_pred.masked_fill_(mask[:, 1:], 0)
        return type_pred, dtimes_pred

    # def decode(self, all_encs):
    #     dist_params = self.time_stack(all_encs)
    #     locs = dist_params[...,:self.num_component]
    #     log_scales = dist_params[...,self.num_component:(2*self.num_component)]
    #     log_weights = dist_params[...,(2*self.num_component):]
    #     log_weights = torch.log_softmax(log_weights, dim=-1)
    #     inter_time_dist = LogNormMix(locs=locs, log_scales=log_scales, log_weights=log_weights)
    #     type_prob = torch.log_softmax(self.mark_stack(all_encs), dim=-1)
    #     type_dist = Categorical(logits=type_prob)
    #     return type_dist, inter_time_dist

    # def compute_loss(self, type_seq, time_seq): # 1-seq_len
    #     device = self.device_indicator.device
    #     type_seq = type_seq.to(device)
    #     time_seq = time_seq.to(device) # 1-seq_len
    #     all_encs = self.encode(type_seq[:, :-1], time_seq[:, :-1]) # 1-seq_len-1

    #     mask = type_seq.eq(0) # 1-seq_len
    #     dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
    #     dtimes.masked_fill_(mask[:, 1:], 0)
    #     # print(dtimes.masked_select(~mask[:, 1:]).mean())
    #     # print(dtimes.masked_select(~mask[:, 1:]).max())
    #     # print(dtimes.masked_select(~mask[:, 1:]).min())
    #     # print(dtimes.masked_select(~mask[:, 1:]).median())
    #     dtimes.clamp_(1e-10)

    #     type_dist, inter_time_dist = self.decode(all_encs) # 2-seq_len

    #     time_log_probs = inter_time_dist.log_prob(dtimes) # 2-seq_len
    #     time_log_probs.masked_fill_(mask[:, 1:], 0)
    #     temp_mark = (type_seq.masked_fill(mask, 1) - 1)[:, 1:] # 2-seq_len
    #     type_log_probs = type_dist.log_prob(temp_mark) # 2-seq_len
    #     type_log_probs.masked_fill_(mask[:, 1:], 0)
    #     # log_probs = time_log_probs + type_log_probs
    #     # log_probs.masked_fill_(mask[:, 1:], 0)
    #     time_loss = -time_log_probs.sum()
    #     type_loss = -type_log_probs.sum()
    #     loss = time_loss + type_loss
    #     return loss, type_loss, time_loss

    # def predict(self, type_seq, time_seq):
    #     device = self.device_indicator.device
    #     type_seq = type_seq.to(device)
    #     time_seq = time_seq.to(device) # 1-seq_len
    #     mask = type_seq.eq(0)
    #     all_encs = self.encode(type_seq, time_seq) # 1-seq_len

    #     type_dist, inter_time_dist = self.decode(all_encs) # 2-seq_len+1
    #     type_logits = type_dist.logits
    #     type_pred = torch.argmax(type_logits, dim=-1) + 1 # (batch_size, seq_len) 2-seq_len+1
    #     # type_pred.masked_fill_(mask, 0)
    #     # time_pred = inter_time_dist.mean # (batch_size, seq_len) 2-seq_len+1
    #     # time_pred.masked_fill_(mask, 0)
    #     # type_pred = type_dist.sample()+1
    #     time_pred = inter_time_dist.sample()
    #     return type_pred, time_pred

    def plotKernel(self):
        self.eval()
        with torch.no_grad():
            device = self.device_indicator.device
            num_layers = len(self.horizon)
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
                plt.locator_params(axis='x', nbins=5)
                for j in range(kern_vals.shape[1]):
                    plt.plot(dt_vals, kern_vals[:, j], linewidth=2)
                # plt.xlabel('Δt', fontsize=32)
                # plt.ylabel('ψ(Δt)', fontsize=32)
                # plt.title(f'layer {i}', fontsize=25)
                plt.xticks(fontsize=32)
                plt.yticks(fontsize=32)
                plt.savefig(f'figs/layer {i}.jpg', bbox_inches='tight')
                plt.clf()
        self.train()
    # def predict(self, type_seq, time_seq):
    #     device = self.device_indicator.device
    #     seq_len = time_seq.shape[0]
    #     type_seq = type_seq.to(device).unsqueeze(0) # (1, seq_len) 1-seq_len
    #     time_seq = time_seq.to(device).unsqueeze(0) # (1, seq_len) 1-seq_len
    #     embed_seq = self.embed(type_seq)
    #     dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
    #     dtimes.clamp_(1e-10)
    #     mask = torch.zeros(1, seq_len).bool().to(device) # 1-seq_len
    #     embed_seq = self.conv(embed_seq, time_seq, mask) # 1-seq_len
    #     temporal = torch.cat([torch.zeros(1, 1, device=device), dtimes], dim=1).log() # 1-seq_len
    #     embed_seq = torch.cat([embed_seq, temporal.unsqueeze(-1)], dim=-1)
    #     self.rnn.flatten_parameters()
    #     all_encs = self.rnn(embed_seq)[0] # (1, seq_len, hidden_dim) 1-seq_len
    #     final_enc = all_encs[0, -1, :] # (hidden_dim)

    #     dist_params = self.time_stack(final_enc) # (3*num_component)
    #     locs = dist_params[:self.num_component]
    #     log_scales = dist_params[self.num_component:(2*self.num_component)]
    #     # log_scales = clamp_preserve_gradients(log_scales, -5.0, 2.0)
    #     log_weights = dist_params[(2*self.num_component):]
    #     log_weights = torch.log_softmax(log_weights, dim=-1)
    #     inter_time_dist = LogNormMix(locs=locs, log_scales=log_scales, 
    #                                 log_weights=log_weights
    #                                 )
    #     dtime_pred = inter_time_dist.mean.item()
    #     # dtime_pred = inter_time_dist.sample().item()
    #     type_prob_logits = torch.log_softmax(self.mark_stack(final_enc), dim=-1) # （num_types)
    #     # type_dist = Categorical(logits=type_prob_logits)
    #     # type_pred = (type_dist.sample()+1).item()
    #     type_pred = (torch.argmax(type_prob_logits)+1).item()
    #     return type_pred, dtime_pred
