import torch
from torch import nn
from .modules.lognorm import LogNormMix
from torch.distributions import Categorical

class LogNormMix(nn.Module):
    def __init__(self, config):
        """
        Required arguments:
            num_types, embed_dim, hidden_dim, num_channel, horizon, num_component,
            # mean_log_inter, std_log_inter
        """
        super(LogNormMix, self).__init__()
        num_types = config['num_types']
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_component = config['num_component']
        self.embed = nn.Embedding(num_types+2, embed_dim, padding_idx=0)
        # mean_log_inter = config['mean_log_inter']
        # std_log_iner = config['std_log_inter']
        siren_dim = config['siren_dim'] if 'siren_dim' in config else 32
        siren_layers = config['siren_layers'] if 'siren_layers' in config else 3
        self.rnn = nn.GRU(input_size=embed_dim+1, hidden_size=hidden_dim, 
                    batch_first=True)
        self.time_stack = nn.Linear(hidden_dim, 3*num_component)
        self.mark_stack = nn.Linear(hidden_dim, num_types)
        self.init_context = nn.Parameter(torch.randn(hidden_dim))
        # self.mean_log_inter = mean_log_inter
        # self.std_log_inter = std_log_iner
        self.num_component = num_component


    def compute_loss(self, type_seq, time_seq):
        device = self.init_context.device
        type_seq = type_seq.to(device)
        embed_seq = self.embed(type_seq)
        time_seq = time_seq.to(device) # 0-seq_len
        batch_size, seq_len = type_seq.size()
        seq_len-=1
        embed_seq = embed_seq[:, 1:, :] # 1-seq_len
        type_seq =type_seq[:, 1:] # 1-seq_len
        mask = type_seq.eq(0) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 1-seq_len
        dtimes.masked_fill_(mask, 0)
        dtimes.clamp_(1e-10)
        embed_seq = embed_seq[:, :-1, :] # (batch_size, seq_len-1, embed_dim) 1-seq_len-1
        temporal = dtimes[:, :-1].log()
        embed_seq = torch.cat([embed_seq, temporal.unsqueeze(-1)], dim=-1)
        self.rnn.flatten_parameters()
        all_encs = self.rnn(embed_seq)[0] # (batch_size, seq_len-1, hidden_dim) 1-seq_len-1
        init_context = self.init_context[None, None, :].expand(batch_size, 1, -1)
        all_encs = torch.cat([init_context, all_encs], dim=1) # 0-seq_len-1

        dist_params = self.time_stack(all_encs)
        locs = dist_params[...,:self.num_component]
        log_scales = dist_params[...,self.num_component:(2*self.num_component)]
        # log_scales = clamp_preserve_gradients(log_scales, -5.0, 2.0)
        log_weights = dist_params[...,(2*self.num_component):]
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormMix(locs=locs, log_scales=log_scales, log_weights=log_weights)
        # time_log_probs: (batch_size, seq_len)
        time_log_probs = inter_time_dist.log_prob(dtimes)
        # type_prob: (batch_size, seq_len, num_type)
        type_prob = torch.log_softmax(self.mark_stack(all_encs), dim=-1)
        type_dist = Categorical(logits=type_prob)
        temp_mark = type_seq.masked_fill(mask, 1) - 1
        type_log_probs = type_dist.log_prob(temp_mark)
        log_probs = time_log_probs + type_log_probs
        log_probs.masked_fill_(mask, 0)
        return -log_probs.sum()

    def predict(self, embed_seq, time_seq):
        device = self.init_context.device
        seq_len = time_seq.shape[0]-1
        embed_seq = embed_seq.to(device).unsqueeze(0) # (1, seq_len+1, embed_dim) 0-seq_len
        time_seq = time_seq.to(device).unsqueeze(0) # (1, seq_len+1) 0-seq_len
        embed_seq = embed_seq[:, 1:, :] # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 1-seq_len
        dtimes.clamp_(1e-10)
        mask = torch.zeros(1, seq_len).bool().to(device)
        temporal = dtimes.log()
        embed_seq = torch.cat([embed_seq, temporal.unsqueeze(-1)], dim=-1)
        self.rnn.flatten_parameters()
        all_encs = self.rnn(embed_seq)[0] # (1, seq_len+1, hidden_dim) 0-seq_len
        final_enc = all_encs[0, -1, :] # (hidden_dim)

        dist_params = self.time_stack(final_enc) # (3*num_component)
        locs = dist_params[:self.num_component]
        log_scales = dist_params[self.num_component:(2*self.num_component)]
        # log_scales = clamp_preserve_gradients(log_scales, -5.0, 2.0)
        log_weights = dist_params[(2*self.num_component):]
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormMix(locs=locs, log_scales=log_scales, 
                                    log_weights=log_weights
                                    )
        dtime_pred = inter_time_dist.mean.item()
        # dtime_pred = inter_time_dist.sample().item()
        type_prob_logits = torch.log_softmax(self.mark_stack(final_enc), dim=-1) # （num_types)
        # type_dist = Categorical(logits=type_prob_logits)
        # type_pred = (type_dist.sample()+1).item()
        type_pred = (torch.argmax(type_prob_logits)+1).item()
        return type_pred, dtime_pred

