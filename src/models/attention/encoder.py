import torch.nn as nn
import torch

from .utils import check_shape

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        dec_hidden_dim,
        dropout=0,
        padding_idx=None,
    ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dec_hidden_dim = dec_hidden_dim
        
        self.embedding = nn.Embedding(
            input_dim,
            embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            bidirectional=True,
        )
        
        self.fc = nn.Linear(2 * hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        num_steps, batch_size = x.shape
        self.num_steps = num_steps
        
        # x.shape: [num_steps, batch_size]
        emb = self.dropout(self.embedding(x))
        # emb.shape: [num_steps, batch_size, emb_dim]
        check_shape(emb, (num_steps, batch_size, self.embed_dim), 'emb')
        
        outputs, hidden = self.rnn(emb)
        # outputs.shape: [num_steps, batch_size, hidden_dim * bidirectional]
        # hidden.shape: [num_layers * bidirectional, batch_size, hidden_dim]
        check_shape(outputs, (num_steps, batch_size, self.hidden_dim * 2), 'outputs')
        check_shape(hidden, (2, batch_size, self.hidden_dim))
        
        hidden = torch.tanh(self.fc(
            torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)
        ))
        check_shape(hidden, (batch_size, self.dec_hidden_dim), 'hidden')
        
        # outputs.shape: [num_steps, batch_size, hidden_dim * bidirectional]
        # hidden.shape: [batch_size, dec_hidden_dim]
        return outputs, hidden