import torch
from torch import nn
from .modules.lognorm import LogNormMix
from torch.distributions import Categorical

class LogNormMixTPP_Pred(nn.Module):
    def __init__(self, config):
        """
        Required arguments:
            num_types, embed_dim, hidden_dim, num_component,
        """
        super(LogNormMixTPP_Pred, self).__init__()
        num_types = config['num_types']
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_component = config['num_component']
        self.embed = nn.Embedding(num_types+1, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=embed_dim+1, hidden_size=hidden_dim, 
                    batch_first=True)
        self.time_stack = nn.Linear(hidden_dim, 3*num_component)
        self.mark_stack = nn.Linear(hidden_dim, num_types)
        self.num_component = num_component
        self.gamma = config['gamma'] if 'gamma' in config else 1
        self.register_buffer('device_indicator', torch.empty(0))

    def encode(self, type_seq, time_seq):
        device = self.device_indicator.device
        embed_seq = self.embed(type_seq)
        batch_size, seq_len = type_seq.size()
        mask = type_seq.eq(0) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
        dtimes.masked_fill_(mask[:, 1:], 0)
        dtimes.clamp_(1e-10)
        temporal = torch.cat([torch.ones(batch_size, 1, device=device)*1e-10, dtimes], dim=1) # TODO: .log() 1-seq_len
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
