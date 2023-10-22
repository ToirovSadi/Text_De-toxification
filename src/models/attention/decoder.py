import torch.nn as nn
import torch

from .utils import check_shape

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embed_dim,
        hidden_dim,
        attention,
        enc_hidden_dim,
        dropout=0,
        padding_idx=None,
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(
            output_dim,
            embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.GRU(
            embed_dim + 2 * enc_hidden_dim,
            hidden_dim,
        )
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(enc_hidden_dim * 2 + hidden_dim + embed_dim, output_dim)
        
    def forward(self, x, hidden, enc_output):
        # x.shape: [batch_size]
        # hidden: [batch_size, dec_hidden_dim]
        # enc_output: [num_steps, batch_size, enc_hidden_dim * 2]
        batch_size = x.shape[0]
        num_steps = enc_output.shape[0]
        num_layers = hidden.shape[0]
        check_shape(hidden, (batch_size, self.hidden_dim), 'hidden')
        check_shape(enc_output, (num_steps, batch_size, self.enc_hidden_dim * 2), 'enc_output')
        
        x = x.unsqueeze(0) # -> x.shape: [1, batch_size]
        check_shape(x, (1, batch_size), 'x')
        
        emb = self.dropout(self.embedding(x))
        # emb.shape: [1, batch_size, embed_dim]
        check_shape(emb, (1, batch_size, self.embed_dim), 'emb')
        
        attn_weights = self.attention(hidden, enc_output).unsqueeze(1)
        # attn_weights.shape: [batch_size, 1, num_steps]
        check_shape(attn_weights, (batch_size, 1, num_steps), 'attn_weights')
        
        enc_output = enc_output.permute(1, 0, 2)
        # enc_output.shape: [batch_size, num_steps, enc_hidden_dim * 2]
        check_shape(enc_output, (batch_size, num_steps, self.enc_hidden_dim * 2), 'enc_output')
        
        attn = torch.bmm(attn_weights, enc_output).squeeze(1)
        # attn.shape: [batch_size, enc_hidden_dim * 2]
        check_shape(attn, (batch_size, self.enc_hidden_dim * 2), 'attn')
        
        attn = attn.unsqueeze(0)
        # attn.shape: [1, batch_size, enc_hidden_dim * 2]
        
        output, hidden = self.rnn(torch.cat((attn, emb), dim=2), hidden.unsqueeze(0))
        # output.shape: [1, batch_size, dec_hidden_dim]
        # hidden.shape: [num_layers, batch_size, dec_hidden_dim]
        check_shape(output, (1, batch_size, self.hidden_dim), 'output')
        check_shape(hidden, (1, batch_size, self.hidden_dim), 'hidden')
        
        # fc_out: takes attn, outputs, emb
        # attn.shape: [1, batch_size, enc_hidden_dim * 2]
        # output.shape: [1, batch_size, dec_hidden_dim]
        # emb.shape: [1, batch_size, embed_dim]
        attn = attn.squeeze(0)
        output = output.squeeze(0)
        emb = emb.squeeze(0)
        
        prediction = self.fc_out(torch.cat((attn, output, emb), dim=1))
        # prediction.shape: [batch_size, output_dim]
        check_shape(prediction, (batch_size, self.output_dim), 'prediction')
        
        return prediction, hidden.squeeze(0)