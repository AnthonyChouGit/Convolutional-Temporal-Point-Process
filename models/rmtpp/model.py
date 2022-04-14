import torch
from torch import nn
from torch.distributions import Categorical

class RMTPP(nn.Module):
    def __init__(self, config) -> None:
        super(RMTPP, self).__init__()
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        num_types = config['num_types']
        mlp_dim = config['mlp_dim']
        max_t = config['max_t'] if 'max_t' in config else 100
        n_samples_pred = config['n_samples_pred'] if 'n_sample_pred' in config else 1000
        self.max_t = max_t
        self.n_samples_pred = n_samples_pred
        # TODO: Change LSTM to the proposed RELU RNN
        self.embed = nn.Embedding(num_types+1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim+1, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=False)
        self.stack = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_types+1) # last dim for time prediction
        )
        self.register_buffer('device_indicator', torch.empty(0))
        self.dtime_w = nn.Parameter(torch.rand(1))

    def encode(self, type_seq, time_seq):
        embed_seq = self.embed(type_seq)
        mask = type_seq.eq(0)
        time_seq.masked_fill_(mask, 0)
        lstm_input = torch.cat([embed_seq, time_seq.unsqueeze(-1)], dim=-1)
        all_encs, _ = self.lstm(lstm_input) # (batch_size, seq_len, hidden_dim) 1-seq_len
        return all_encs

    def compute_loss(self, type_seq, time_seq):
        device = self.device_indicator.device
        dtimes = (time_seq[:, 1:] - time_seq[:, :-1]).to(device).unsqueeze(-1) # 2-seq_len
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device)# 1-seq_len
        mask = type_seq.eq(0)
        all_encs = self.encode(type_seq[:, :-1], time_seq[:, :-1]) # 1-seq_len-1
        assert not torch.any(all_encs.isnan())
        assert not torch.any(all_encs.isinf())
        all_encs = self.stack(all_encs)
        type_encs = all_encs[:, :, :-1]
        time_enc = all_encs[:, :, -1].unsqueeze(-1)
        prob_all_types = torch.log_softmax(type_encs, dim=-1) # (batch_size, seq_len, num_types) 2-seq_len
        type_dists = Categorical(logits=prob_all_types) # 2-seq_len
        temp_types = (type_seq.masked_fill(mask, 1) - 1)[:, 1:] # 2-seq_len
        type_log_prob = type_dists.log_prob(temp_types)
        type_log_prob.masked_fill_(mask[:, 1:], 0)
        ewt = -torch.exp(self.dtime_w)
        eiwt = -torch.exp(-self.dtime_w)
        time_log_prob = time_enc + ewt * dtimes  +\
         (torch.exp(time_enc)-torch.exp(time_enc + ewt * dtimes ))\
              * eiwt
        assert not torch.any(time_log_prob.isnan())
        assert not torch.any(time_log_prob.isinf())

        time_log_prob = time_log_prob.squeeze(-1) # (batch_size, seq_len)
        time_log_prob.masked_fill_(mask[:, 1:], 0) # 2-seq_len
        nll = -torch.sum(type_log_prob+time_log_prob)
        assert not nll.isnan()
        assert not nll.isinf()
        return nll

    def predict(self, type_seq, time_seq):
        device = self.device_indicator.device
        dtimes = (time_seq[:, 1:] - time_seq[:, :-1]).to(device).unsqueeze(-1) # 2-seq_len
        type_seq = type_seq.to(device)
        time_seq = time_seq.to(device)# 1-seq_len
        mask = type_seq.eq(0)
        all_encs = self.encode(type_seq, time_seq)
        all_encs = self.stack(all_encs)
        type_encs = all_encs[:, :, :-1]
        time_enc = all_encs[:, :, -1].unsqueeze(-1) # (batch_size, seq_len, 1)
        estimated_type = torch.argmax(type_encs, dim=-1) + 1
        max_t = self.max_t
        n_samples = self.n_samples_pred
        dt_vals = torch.linspace(0, max_t, n_samples+1).to(device) # (n_samples+1)
        dt_vals = dt_vals.view(1, 1, n_samples+1, 1)
        time_enc = time_enc[:, :, None, :]
        density = torch.exp(time_enc + self.dtime_w * dt_vals + self.dtime_b +\
         (torch.exp(time_enc+self.dtime_b)-torch.exp(time_enc + self.dtime_w * dt_vals + self.dtime_b))\
              / self.dtime_w) # (batch_size, seq_len, n_samples+1, 1)
        time_step = max_t / n_samples
        t_ft = (density * dt_vals).squeeze(-1) # (batch_size, seq_len, n_samples+1)
        estimated_dt = torch.sum(0.5 * (t_ft[:, :, 1:]+t_ft[:, :, :-1]) * time_step, dim=-1)
        return estimated_type, estimated_dt


    # def predict(self, type_seq, time_seq):
    #     device = self.device_indicator.device
    #     # dtimes = (time_seq[:, 1:] - time_seq[:, :-1]).to(device).unsqueeze(-1)
    #     type_seq = type_seq[:, 1:].to(device)
    #     time_seq = time_seq[:, 1:].to(device)
    #     embed_seq = self.embed(type_seq)
    #     batch_size, seq_len = type_seq.size()
    #     lstm_input = torch.cat([embed_seq, time_seq.unsqueeze(-1)], dim=-1)
    #     all_encs, _ = self.lstm(lstm_input) # (1, seq_len, hidden_dim) 1-seq_len
    #     final_enc = all_encs[:, -1, :]
    #     final_enc = self.stack(final_enc)
    #     type_enc = final_enc[:, :, :-1]
    #     time_enc = final_enc[:, :, -1].unsqueeze(-1) # (1, 1)
    #     type_pred = torch.argmax(type_enc, dim=-1).item() + 1
    #     max_t = self.max_t
    #     n_samples = self.n_sample_pred
    #     dt_vals = torch.linspace(0, max_t, n_samples+1).to(device).view(1, -1, 1) # (1, n_samples+1, 1)
    #     dt_vals = dt_vals.squeeze(0).squeeze(-1) # n_samples+1
    #     time_enc = time_enc.flatten() # 1
    #     density = torch.exp(time_enc + self.dtime_w * dt_vals + self.dtime_b +\
    #      (torch.exp(time_enc+self.dtime_b)-torch.exp(time_enc + self.dtime_w * dt_vals + self.dtime_b))\
    #           / self.dtime_w) # n_samples+1
    #     time_step = max_t / n_samples
    #     t_ft = density * dt_vals.flatten() # (n_samples+1)
    #     estimated_dt = torch.sum(0.5 * (t_ft[1:] + t_ft[:-1]) * time_step).item()
    #     return type_pred, estimated_dt