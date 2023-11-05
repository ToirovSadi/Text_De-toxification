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
        
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        
        self.fc = nn.Linear(2 * hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_steps = x.shape
        
        # x.shape: [batch_size, num_steps]
        emb = self.dropout(self.embedding(x))
        # emb.shape: [batch_size, num_steps, emb_dim]
        check_shape(emb, (batch_size, num_steps, self.embed_dim), 'emb')
        
        outputs, hidden = self.rnn(emb)
        # outputs.shape: [batch_size, num_steps, hidden_dim * bidirectional]
        # hidden.shape: [num_layers * bidirectional, batch_size, hidden_dim]
        check_shape(outputs, (batch_size, num_steps, self.hidden_dim * 2), 'outputs')
        check_shape(hidden, (2, batch_size, self.hidden_dim))
        
        hidden = torch.tanh(self.fc(
            torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)
        ))
        check_shape(hidden, (batch_size, self.dec_hidden_dim), 'hidden')
        
        # outputs.shape: [batch_size, num_steps, hidden_dim * bidirectional]
        # hidden.shape: [batch_size, dec_hidden_dim]
        return outputs, hidden