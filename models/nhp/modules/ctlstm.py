import torch
from torch import nn

class CTLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(CTLSTM, self).__init__()
        self.trans_stack = nn.Sequential(
            nn.Linear(embed_dim+hidden_dim, hidden_dim*7),
            nn.Dropout(0.2)
        )
        self.softplus = nn.Softplus()
        self.register_buffer('device_indicator', torch.empty(0))
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

    def forward(self, embeds, dtimes):
        """
        Args:
            embeds (batch_size, max_len , embed_dim): input embeddings
            dtimes(batch_size, max_len): Time intervals after each event given by embeds
        inputs: 1 -- seq_len-1
        d_times: 2 -- seq_len
        updated_states: 1 -- seq_len-1
        decayed_hs: 2 -- seq_len
        """
        batch_size, max_len, embed_dim = embeds.size()
        device = self.device_indicator.device
        decayed_h = torch.zeros(batch_size, self.hidden_dim, device=device)
        decayed_c = torch.zeros(batch_size, self.hidden_dim, device=device)
        last_cbar = torch.zeros(batch_size, self.hidden_dim, device=device)
        updated_cells = list()
        updated_cell_bars = list()
        updated_deltas = list()
        updated_output_gates = list()
        decayed_hs = list()
        for i in range(max_len):
            input = embeds[:, i, :] # (batch_size, embed_dim)
            input_ht = torch.cat([input, decayed_h], dim=-1)
            transformed = self.trans_stack(input_ht) # (batch_size, 7*hidden_dim)
            sigmoid_trans = torch.sigmoid(transformed[:, :6*self.hidden_dim])
            softplus_trans = self.softplus(transformed[:, 6*self.hidden_dim:])
            input_gate = sigmoid_trans[:, :self.hidden_dim]
            input_gate_bar = sigmoid_trans[:, self.hidden_dim:2*self.hidden_dim]
            forget_gate = sigmoid_trans[:, 2*self.hidden_dim:3*self.hidden_dim]
            forget_gate_bar = sigmoid_trans[:, 3*self.hidden_dim:4*self.hidden_dim]
            output_gate = sigmoid_trans[:, 4*self.hidden_dim:5*self.hidden_dim]
            trans_input = 2*sigmoid_trans[:, 5*self.hidden_dim:]-1
            delta = softplus_trans
            cell = forget_gate*decayed_c + input_gate*trans_input
            cell_bar = forget_gate_bar*last_cbar + input_gate_bar*trans_input
            updated_cells.append(cell)
            updated_cell_bars.append(cell_bar)
            updated_deltas.append(delta)
            updated_output_gates.append(output_gate)
            dtime = dtimes[:, i].view(-1, 1)
            decayed_h, decayed_c = decay(cell, cell_bar, delta, output_gate, dtime)
            last_cbar = cell_bar
            decayed_hs.append(decayed_h)
        updated_states = (updated_cells, updated_cell_bars, updated_deltas, updated_output_gates)
        return updated_states, decayed_hs


def decay(cell, cell_bar, delta, output_gate, d_time):
    """
    Args:
        cell (batch_size,*, hidden_dim): 
        cell_bar (batch_size,*, hidden_dim): 
        delta (batch_size,*, hidden_dim): 
        output_gate (batch_size,*, hidden_dim): 
        d_time (batch_size,*, 1): 
    """
    decayed_c = cell_bar + (cell - cell_bar)*torch.exp(-delta*d_time)
    decayed_h = output_gate * (2 * torch.sigmoid(2 * decayed_c) - 1)
    return decayed_h, decayed_c