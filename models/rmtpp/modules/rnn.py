from turtle import forward
import torch
from torch import nn

class ReluRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(ReluRNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(embed_dim+hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.register_buffer('device_indicator', torch.empty(0))
        self.hidden_dim = hidden_dim

    def forward(self, embed_seq):
        device = self.device_indicator.device
        batch_size, seq_len, embed_dim = embed_seq.size()
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_list = list()
        for i in range(seq_len):
            embed = embed_seq[:, i, :] # (batch_size, embed_dim)
            embed_hidden = torch.cat([embed, hidden], dim=-1)
            hidden = self.stack(embed_hidden)
            hidden_list.append(hidden)
        encs = torch.stack(hidden_list, dim=1)
        return encs