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
        vocab=None,
        padding_idx=None,
    ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.vocab = vocab
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            input_dim,
            embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        
        self.fc_hidden = nn.Linear(2 * hidden_dim, dec_hidden_dim)
        self.fc_cell = nn.Linear(2 * hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_steps = x.shape
        
        # x.shape: [batch_size, num_steps]
        emb = self.dropout(self.embedding(x))
        # emb.shape: [batch_size, num_steps, emb_dim]
        check_shape(emb, (batch_size, num_steps, self.embed_dim), 'emb')
        
        outputs, state = self.rnn(emb)
        # outputs.shape: [batch_size, num_steps, hidden_dim * bidirectional]
        check_shape(outputs, (batch_size, num_steps, self.hidden_dim * 2), 'outputs')
        
        hidden = torch.tanh(self.fc_hidden(
            torch.cat((state[0][-1, :, :], state[0][-2, :, :]), dim=1)
        )).unsqueeze(0)
        check_shape(hidden, (1, batch_size, self.dec_hidden_dim), 'hidden')
        
        cell = torch.tanh(self.fc_cell(
            torch.cat((state[1][-1, :, :], state[1][-2, :, :]), dim=1)
        )).unsqueeze(0)
        check_shape(cell, (1, batch_size, self.dec_hidden_dim), 'cell')
        
        # outputs.shape: [batch_size, num_steps, hidden_dim * bidirectional]
        return outputs, (hidden, cell)