import torch
from torch import nn
from .modules.ctlstm import CTLSTM, decay

class NHP(nn.Module):
    def __init__(self, config):
        """
        Required arguments:
            embed_dim, hidden_dim, num_types
        """
        super(NHP, self).__init__()
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_types = config['num_types']
        n_sample_train = config['n_sample_train'] if 'n_sample_train' in config else 3
        n_sample_eval = config['n_sample_eval'] if 'n_sample_eval' in config else 10
        n_sample_pred = config['n_sample_pred'] if 'n_sample_pred' in config else 1000
        max_t = config['max_t'] if 'max_t' in config else 100
        self.embed = nn.Embedding(num_types+2, embed_dim, padding_idx=0)
        self.ctlstm = CTLSTM(embed_dim, hidden_dim)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.n_sample_train = n_sample_train
        self.n_sample_eval = n_sample_eval
        self.max_t = max_t
        self.n_sample_pred = n_sample_pred
        self.register_buffer('device_indicator', torch.empty(0))

        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_types, bias=False),
            nn.Softplus()
        )

    def compute_loss(self, type_seq, time_seq):
        """
        Args:
            embed_seq (batch_size, max_len+1, embed_dim): 0-seq_len
            type_seq (batch_size, max_len+1): 0-seq_len
            time_seq (batch_size, max_len+1): 0-seq_len
        """
        batch_size, max_len = time_seq.size()
        max_len-=1
        device = self.device_indicator.device
        if self.training:
            n_sample = self.n_sample_train
        else:
            n_sample = self.n_sample_eval
        # embed_seq = embed_seq.to(device)
        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        embed_seq = self.embed(type_seq)
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # (batch_size, max_len) 1-seq_len
        mask = type_seq.eq(0)
        dtimes.masked_fill_(mask[:, 1:], 0)
        updated_states, decayed_hs = self.ctlstm(embed_seq[:, :-1, :], dtimes)
        # decayed_hs: (batch_size, max_len, hidden_dim) 1 -- seq_len
        decayed_hs = torch.stack(decayed_hs, dim=1)
        type_intensities = self.intensity_layer(decayed_hs) # (batch_size, max_len, num_types)
        type_seq = type_seq[:, 1:] # (batch_size, max_len) 1-seq_len
        type_intensities = torch.cat([torch.ones(batch_size, max_len, 1, device=device), type_intensities], dim=-1)
        event_intensities = type_intensities[torch.arange(batch_size)[:,None], 
                    torch.arange(max_len)[None,:], type_seq] # (batch_size, max_len)
        event_intensities.clamp_(1e-10)
        log_intensities = torch.log(event_intensities)
        log_intensities.masked_fill_(mask[:, 1:], 0)
        assert not torch.any(log_intensities.isnan())
        assert not torch.any(log_intensities.isinf())

        dtime_samples = torch.rand(batch_size, max_len, n_sample, device=device) * dtimes[:, :, None]
        updated_cells, updated_cell_bars, updated_deltas, updated_output_gates = updated_states
        # lists of (batch_size, hidden_dim)
        # (batch_size, max_len, samples_per_interval, hidden_dim) 0 -- seq_len-1
        updated_cells = torch.stack(updated_cells, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_cell_bars = torch.stack(updated_cell_bars, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_deltas = torch.stack(updated_deltas, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_output_gates = torch.stack(updated_output_gates, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        sample_decayed_hs, _ = decay(updated_cells, updated_cell_bars, updated_deltas, updated_output_gates, dtime_samples.unsqueeze(-1))
        # (batch_size, max_len, sample_per_intervals, num_types)
        sample_intensities = self.intensity_layer(sample_decayed_hs)
        integrals = sample_intensities.sum(dim=3).mean(dim=2) * dtimes
        integrals.masked_fill_(mask[:, 1:], 0)
        assert not torch.any(torch.isnan(integrals))
        assert not torch.any(torch.isinf(integrals))
        nll = torch.sum(integrals) - torch.sum(log_intensities)

        return nll

    def predict(self, type_seq, time_seq):
        """
            embed_seq: (max_len+1, embed_dim) 0-seq_len
            time_seq: (max_len+1) 0-seq_len
        """
        device = self.device_indicator.device
        n_samples = self.n_sample_pred
        max_t = self.max_t
        embed_seq = self.embed(type_seq.to(device)).unsqueeze(0)
        time_seq = time_seq.to(device).unsqueeze(0)
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # (1, max_len) 1-seq_len
        dtimes = torch.cat([dtimes, torch.zeros(1, 1, device=device)], dim=-1) # 1-seq_len+1
        updated_states, _ = self.ctlstm(embed_seq, dtimes)
        updated_cells, updated_cell_bars, updated_deltas, updated_output_gates = updated_states
        # (1, n_samples+1, hidden_dim)
        last_cell = updated_cells[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        last_cell_bar = updated_cell_bars[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        last_delta = updated_deltas[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        last_output_gate = updated_output_gates[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        dt_vals = torch.linspace(0, max_t, n_samples+1).to(device).view(1, -1, 1) # (1, n_samples+1, 1)
        # sample_hts: (1, n_samples+1, hidden_dim)
        sample_hts, _ = decay(last_cell, last_cell_bar, last_delta, last_output_gate, dt_vals)
        type_intensities = self.intensity_layer(sample_hts) # (1, n_samples+1, num_types)
        time_step = max_t / n_samples
        typesum_int = type_intensities.sum(dim=-1).flatten() # (n_samples+1)
        interval_integral = 0.5 * (typesum_int[:-1] + typesum_int[1:]) * time_step # (n_sample)
        cum_integral = torch.cumsum(interval_integral, dim=0)
        # cum_integral: (n_samples+1)
        cum_integral = torch.cat([torch.zeros(1, device=device), cum_integral])
        density = typesum_int * torch.exp(-cum_integral) # (n_samples+1)
        t_ft = density * dt_vals.flatten() # (n_samples+1)
        estimated_dt = torch.sum(0.5 * (t_ft[1:] + t_ft[:-1]) * time_step)
        ratio = type_intensities.squeeze(0) / typesum_int[:, None] # (n_samples+1, num_types)
        type_density = ratio * density[:, None]
        estimate_type_prob = (0.5 * (type_density[1:] + type_density[:-1]) * time_step).sum(dim=0)
        estimate_type = torch.argmax(estimate_type_prob) + 1
        return estimate_type.item(), estimated_dt.item()

