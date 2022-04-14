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
        n_samples_train = config['n_samples_train'] if 'n_samples_train' in config else 3
        n_samples_eval = config['n_samples_eval'] if 'n_samples_eval' in config else 10
        n_samples_pred = config['n_samples_pred'] if 'n_samples_pred' in config else 1000
        max_t = config['max_t'] if 'max_t' in config else 100
        self.embed = nn.Embedding(num_types+1, embed_dim, padding_idx=0)
        self.ctlstm = CTLSTM(embed_dim, hidden_dim)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.n_samples_train = n_samples_train
        self.n_samples_eval = n_samples_eval
        self.max_t = max_t
        self.n_samples_pred = n_samples_pred
        self.register_buffer('device_indicator', torch.empty(0))

        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_types, bias=False),
            nn.Softplus()
        )

    def compute_loss(self, type_seq, time_seq):
        """
        Args:
            embed_seq (batch_size, max_len+1, embed_dim): 1-seq_len
            type_seq (batch_size, max_len+1): 1-seq_len
            time_seq (batch_size, max_len+1): 1-seq_len
        """
        batch_size, max_len = time_seq.size()
        device = self.device_indicator.device
        if self.training:
            n_sample = self.n_samples_train
        else:
            n_sample = self.n_samples_eval
        # embed_seq = embed_seq.to(device)
        time_seq = time_seq.to(device)
        type_seq = type_seq.to(device)
        embed_seq = self.embed(type_seq)
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # (batch_size, max_len) 2-seq_len
        mask = type_seq.eq(0) # 1-seq_len
        dtimes.masked_fill_(mask[:, 1:], 0)
        updated_states, decayed_hs = self.ctlstm(embed_seq[:, :-1, :], dtimes) # 1-seq_len-1, 2-seq_len
        # decayed_hs: (batch_size, max_len, hidden_dim) 1 -- seq_len
        decayed_hs = torch.stack(decayed_hs, dim=1) # 2-seq_len
        type_intensities = self.intensity_layer(decayed_hs) # 2-seq_len
        type_seq = type_seq[:, 1:] # (batch_size, max_len) 2-seq_len
        type_intensities = torch.cat([torch.ones(batch_size, max_len, 1, device=device), type_intensities], dim=-1)
        event_intensities = type_intensities[torch.arange(batch_size)[:,None], 
                    torch.arange(max_len)[None,:], type_seq] # (batch_size, max_len)
        event_intensities.clamp_(1e-10)
        log_intensities = torch.log(event_intensities) # 2-seq_len
        log_intensities.masked_fill_(mask[:, 1:], 0)
        assert not torch.any(log_intensities.isnan())
        assert not torch.any(log_intensities.isinf())

        dtime_samples = torch.rand(batch_size, max_len-1, n_sample, device=device) * dtimes[:, :, None]
        updated_cells, updated_cell_bars, updated_deltas, updated_output_gates = updated_states # 1-seq_len-1
        # lists of (batch_size, hidden_dim)
        # (batch_size, max_len-1, samples_per_interval, hidden_dim) 1 -- seq_len-1
        updated_cells = torch.stack(updated_cells, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_cell_bars = torch.stack(updated_cell_bars, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_deltas = torch.stack(updated_deltas, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        updated_output_gates = torch.stack(updated_output_gates, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample, self.hidden_dim)
        sample_decayed_hs, _ = decay(updated_cells, updated_cell_bars, updated_deltas, updated_output_gates, dtime_samples.unsqueeze(-1))
        # (batch_size, max_len-1, sample_per_intervals, num_types)
        sample_intensities = self.intensity_layer(sample_decayed_hs) # 1-seq_len-1
        integrals = sample_intensities.sum(dim=3).mean(dim=2) * dtimes
        integrals.masked_fill_(mask[:, 1:], 0)
        assert not torch.any(torch.isnan(integrals))
        assert not torch.any(torch.isinf(integrals))
        nll = torch.sum(integrals) - torch.sum(log_intensities)

        return nll

    def predict(self, type_seq, time_seq):
        """
            type_seq: (batch_size, max_len, embed_dim) 1-seq_len
            time_seq: (batch_size, max_len) 1-seq_len

            prediction result: 2-seq_len+1
        """
        device = self.device_indicator.device
        n_sample = self.n_samples_pred
        batch_size, max_len = type_seq.size()
        max_t = self.max_t
        embed_seq = self.embed(type_seq.to(device)) # 1-seq_len
        time_seq = time_seq.to(device) # 1-seq_len
        dtimes = time_seq[:, 1:] - time_seq[:, :-1] # 2-seq_len
        dtimes = torch.cat([dtimes, torch.zeros(batch_size, 1, device=device)], dim=-1) # 2-seq_len+1
        updated_states, _ = self.ctlstm(embed_seq, dtimes) # 1-seq_len
        # (batch_size, seq_len, hidden_dim) 1-seq_len
        updated_cells, updated_cell_bars, updated_deltas, updated_output_gates = updated_states
        updated_cells = torch.stack(updated_cells, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample+1, self.hidden_dim)
        updated_cell_bars = torch.stack(updated_cell_bars, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample+1, self.hidden_dim)
        updated_deltas = torch.stack(updated_deltas, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample+1, self.hidden_dim)
        updated_output_gates = torch.stack(updated_output_gates, dim=1)[:, :, None, :]\
                .expand(batch_size, max_len, n_sample+1, self.hidden_dim)
        dt_vals = torch.linspace(0, max_t, n_sample+1).to(device) # n_sample+1
        time_step = max_t / n_sample
        dt_vals = dt_vals.view(1, 1, n_sample+1, 1)
        sample_hts, _ = decay(updated_cells, updated_cell_bars, updated_deltas, updated_output_gates, dt_vals)
        type_intensities = self.intensity_layer(sample_hts) # (batch_size, seq_len, n_sample+1, num_types)
        type_sum_intensities = type_intensities.sum(-1) # (batch_size, seq_len, n_sample+1)
        # (batch_size, seq_len, n_sample)
        interval_integral = 0.5 * (type_sum_intensities[:, :, :-1]+type_sum_intensities[:, :, 1:]) * time_step
        cum_intensity = torch.cumsum(interval_integral, dim=-1)
        # (batch_size, seq_len, n_sample+1)
        cum_intensity = torch.cat([torch.zeros(batch_size, max_len, 1, device=device), cum_intensity], dim=-1)
        density = type_sum_intensities * torch.exp(-cum_intensity) # (batch_size, seq_len, n_sample+1)
        dt_vals = dt_vals.squeeze(-1) # (1, 1, n_sample+1)
        t_ft = density * dt_vals # (batch_size, seq_len, n_sample+1)
        estimated_dt = torch.sum(0.5 * (t_ft[:, :, 1:]+t_ft[:, :, :-1]) * time_step, dim=-1)
        ratio = type_intensities / type_sum_intensities.unsqueeze(-1) # (batch_size, seq_len, n_sample+1, num_types)
        type_density = ratio * density.unsqueeze(-1)
        type_prob = torch.sum(0.5 * (type_density[:, :, 1:, :]+type_density[:, :, :-1, :]) * time_step, dim=-2)
        estimated_type = torch.argmax(type_prob, dim=-1) + 1
        return estimated_type, estimated_dt
        
        # (1, n_samples+1, hidden_dim)
        # last_cell = updated_cells[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        # last_cell_bar = updated_cell_bars[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        # last_delta = updated_deltas[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        # last_output_gate = updated_output_gates[-1][:, None, :].expand(1, n_samples+1, self.hidden_dim)
        # dt_vals = torch.linspace(0, max_t, n_samples+1).to(device).view(1, -1, 1) # (1, n_samples+1, 1)
        # # sample_hts: (1, n_samples+1, hidden_dim)
        # sample_hts, _ = decay(last_cell, last_cell_bar, last_delta, last_output_gate, dt_vals)
        # type_intensities = self.intensity_layer(sample_hts) # (1, n_samples+1, num_types)
        # time_step = max_t / n_samples
        # typesum_int = type_intensities.sum(dim=-1).flatten() # (n_samples+1)
        # interval_integral = 0.5 * (typesum_int[:-1] + typesum_int[1:]) * time_step # (n_sample)
        # cum_integral = torch.cumsum(interval_integral, dim=0)
        # # cum_integral: (n_samples+1)
        # cum_integral = torch.cat([torch.zeros(1, device=device), cum_integral])
        # density = typesum_int * torch.exp(-cum_integral) # (n_samples+1)
        # t_ft = density * dt_vals.flatten() # (n_samples+1)
        # estimated_dt = torch.sum(0.5 * (t_ft[1:] + t_ft[:-1]) * time_step)
        # ratio = type_intensities.squeeze(0) / typesum_int[:, None] # (n_samples+1, num_types)
        # type_density = ratio * density[:, None]
        # estimate_type_prob = (0.5 * (type_density[1:] + type_density[:-1]) * time_step).sum(dim=0)
        # estimate_type = torch.argmax(estimate_type_prob) + 1
        # return estimate_type.item(), estimated_dt.item()

