import torch.nn as nn
import torch

from .utils import check_shape

class Attention(nn.Module):
    def __init__(
        self, 
        enc_hidden_dim,
        dec_hidden_dim,
    ):
        super(Attention, self).__init__()
        
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, enc_outputs):
        # hidden.shape: [batch_size, dec_hidden_dim]
        # enc_outputs.shape: [batch_size, num_steps, enc_hidden_dim * 2]
        
        batch_size, num_steps = enc_outputs.shape[:2]
        
        hidden = hidden.unsqueeze(1).repeat(1, num_steps, 1)
        # hidden.shape: [batch_size, num_steps, dec_hidden_dim]
        check_shape(hidden, (batch_size, num_steps, self.dec_hidden_dim), 'hidden')
        
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        # energy.shape: [batch_size, num_steps, dec_hidden_dim]
        check_shape(energy, (batch_size, num_steps, self.dec_hidden_dim))
        
        energy = self.v(energy).squeeze(2)
        # energy.shape: [batch_size, num_steps]
        check_shape(energy, (batch_size, num_steps), 'energy')
        
        output = torch.softmax(energy, dim=1)
        # output.shape : [batch_size, num_steps]
        check_shape(output, (batch_size, num_steps), 'output')
        return output