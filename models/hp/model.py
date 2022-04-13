import torch
from torch import nn
from .utils.mask import get_subsequent_mask

class HP(nn.Module):
    def __init__(self, config):
        super(HP, self).__init__()
        num_types = config['num_types']
        max_t = config['max_t'] if 'max_t' in config else 100
        n_samples_pred = config['n_samples_pred'] if 'n_samples_pred' in config else 1000
        self.max_t = max_t
        self.n_samples_pred = n_samples_pred
        self.num_types = num_types
        self.alpha = nn.Parameter(torch.randn(num_types+1, num_types+2))
        self.beta = nn.Parameter(torch.randn(1))
        self.mu = nn.Parameter(torch.randn(num_types+1))
        self.softplus = nn.Softplus()

    def compute_loss(self, type_seq, time_seq):
        device = self.alpha.device
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device)
        mhat = self.softplus(self.mu) # (num_types+1)
        Ahat = self.softplus(self.alpha) # (num_types+1, num_types+2)
        omega = self.softplus(self.beta) # (1)
        batch_size, seq_len = type_seq.size()
        seq_len -= 1
        event_mus = mhat[type_seq[:, 1:]] # 1-seq_len (batch_size, seq_len)
        dt = time_seq[:, :, None] - time_seq[:, None, :] # (batch_size, seq_len+1, seq_len+1)
        sub_mask = get_subsequent_mask(seq_len+1).to(device) # (seq_len+1, seq_len+1)
        mask = type_seq.eq(0)
        dt.masked_fill_(sub_mask, 0) 
        dt.masked_fill_(mask[:, :, None], 0)
        dt.masked_fill_(mask[:, None, :], 0)
        dt = dt[:, 1:, :] # (batch_size, seq_len, seq_len+1)
        kern = omega * torch.exp(-omega * dt) # (batch_size, seq_len, seq_len+1)
        rowidx = type_seq[:, 1:, None].expand(batch_size, seq_len, seq_len+1) # (batch_size, seq_len, seq_len+1)
        colidx = type_seq[:, None, :].expand(batch_size, seq_len, seq_len+1)
        alpha = Ahat[rowidx, colidx] # (batch_size, seq_len, seq_len+1)
        ag = alpha * kern # (batch_size, seq_len, seq_len+1)
        ag.masked_fill_(sub_mask[1:, :], 0)
        ag.masked_fill_(mask[:, 1:, None], 0)
        ag.masked_fill_(mask[:, None, :], 0)
        ag = ag.sum(dim=2) # (batch_size, seq_len)
        rates = event_mus + ag
        rates.masked_fill_(mask[:, 1:], 1)
        rates.clamp_(1e-10)
        max_t = time_seq.max(dim=1)[0] # (batch_size) 
        compensator_baseline = max_t * torch.sum(mhat[1:]) # (batch_size)
        log_kernel = -omega * (max_t[:, None] - time_seq) # (batch_size, seq_len+1)
        int_kernel = 1-torch.exp(log_kernel) # (batch_size, seq_len+1)
        au = Ahat[:, type_seq].transpose(0, 1) # (batch_size, num_types+1, seq_len+1)
        au = au[:, 1:, :]
        au_int_kernel = (au * int_kernel.unsqueeze(1)).sum(1) # (batch_size, seq_len+1)
        au_int_kernel.masked_fill_(mask, 0)
        compensator = compensator_baseline + au_int_kernel.sum(1)
        nll = compensator.sum() - torch.log(rates).sum()
        return nll

    def predict(self, type_seq, time_seq):
        device = self.alpha.device
        mhat = self.softplus(self.mu) # (num_types+1)
        Ahat = self.softplus(self.alpha) # (num_types+1, num_types+2)
        omega = self.softplus(self.beta) # (1)
        type_seq = type_seq.to(device) # (seq_len+1)
        time_seq = time_seq.to(device) # (seq_len+1)
        last_t = time_seq[-1]
        max_t = self.max_t
        n_samples = self.n_samples_pred
        dt_vals = torch.linspace(0, max_t, n_samples+1).to(device) # (n_samples+1)
        t_vals = last_t + dt_vals
        t_diff = t_vals[:, None] - time_seq[None, :] # (n_samples+1, seq_len+1)
        kern = omega * torch.exp(-omega * t_diff) # (n_samples+1, seq_len+1)
        auu = Ahat[:, type_seq] # (num_types+1, seq_len+1)
        ag = kern[:, None, :] * auu[None, :, :] # (n_samples+1, num_types+1, seq_len+1)
        ag = ag.sum(dim=-1) # (n_samples+1, num_types+1)
        rates = ag + mhat
        rates = rates[:, 1:] # (n_samples+1, num_types)
        type_sum_rates = rates.sum(1) # (n_samples+1)
        int_base = mhat[None, :] * dt_vals[:, None] # (n_samples+1, num_types+1)
        # int_kernel: (n_samples+1, num_types, seq_len+1)
        int_exp_1 = torch.exp(-omega*(last_t-time_seq)) # (seq_len+1)
        int_exp_2 = torch.exp(-omega*t_diff) # (n_samples+1, seq_len+1)
        int_kernel = int_exp_1[None, :] - int_exp_2 # (n_samples+1, seq_len+1)
        int_ag = int_kernel[:, None, :] * auu[None, :, :] # (n_samples+1, num_types+1, seq_len+1)
        int_ag = int_ag.sum(-1) # (n_samples+1, num_types+1)
        int_rates = int_base + int_ag
        int_rates = int_rates[:, 1:]
        type_sum_int_rates = int_rates.sum(-1) # n_samples+1
        type_sum_probs = type_sum_rates * torch.exp(-type_sum_int_rates) # (n_samples+1)
        t_ft = dt_vals * type_sum_probs
        time_step = max_t / n_samples
        expected_dtime = ((t_ft[1:]+t_ft[:-1]) * 0.5 * time_step).sum()
        ratio = rates / type_sum_rates[:, None] # (n_samples+1, num_types)
        type_density = ratio * type_sum_probs[:, None]
        estimate_type_prob = (0.5 * (type_density[1:] + type_density[:-1]) * time_step).sum(dim=0)
        estimate_type = torch.argmax(estimate_type_prob) + 1
        return estimate_type.item(), expected_dtime.item()

