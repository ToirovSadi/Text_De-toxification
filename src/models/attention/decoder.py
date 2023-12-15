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
        vocab=None,
        padding_idx=None,
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.embed_dim = embed_dim
        self.vocab = vocab
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            output_dim,
            embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.LSTM(
            embed_dim + 2 * enc_hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(enc_hidden_dim * 2 + hidden_dim + embed_dim, output_dim)
        
    def forward(self, x, state, enc_output):
        # x.shape: [batch_size]
        # hidden: [batch_size, dec_hidden_dim]
        # enc_output: [batch_size, num_steps, enc_hidden_dim * 2]
        batch_size = x.shape[0]
        num_steps = enc_output.shape[1]
        check_shape(enc_output, (batch_size, num_steps, self.enc_hidden_dim * 2), 'enc_output')
        
        x = x.unsqueeze(1) # -> x.shape: [batch_size, 1]
        check_shape(x, (batch_size, 1), 'x')
        
        emb = self.dropout(self.embedding(x))
        # emb.shape: [batch_size, 1, embed_dim]
        check_shape(emb, (batch_size, 1, self.embed_dim), 'emb')
        
        attn_weights = self.attention(state[0].squeeze(0), enc_output).unsqueeze(1)
        # attn_weights.shape: [batch_size, 1, num_steps]
        check_shape(attn_weights, (batch_size, 1, num_steps), 'attn_weights')
        
        attn = torch.bmm(attn_weights, enc_output).squeeze(1)
        # attn.shape: [batch_size, enc_hidden_dim * 2]
        check_shape(attn, (batch_size, self.enc_hidden_dim * 2), 'attn')
        
        attn = attn.unsqueeze(1)
        # attn.shape: [batch_size, 1, enc_hidden_dim * 2]
        
        output, state = self.rnn(torch.cat((attn, emb), dim=2), state)
        # output.shape: [batch_size, 1, dec_hidden_dim]
        check_shape(output, (batch_size, 1, self.hidden_dim), 'output')
        
        # fc_out: takes attn, outputs, emb
        # attn.shape: [batch_size, 1, enc_hidden_dim * 2]
        # output.shape: [batch_size, 1, dec_hidden_dim]
        # emb.shape: [batch_size, 1, embed_dim]
        attn = attn.squeeze(1)
        output = output.squeeze(1)
        emb = emb.squeeze(1)
        
        prediction = self.fc_out(torch.cat((attn, output, emb), dim=1))
        # prediction.shape: [batch_size, output_dim]
        check_shape(prediction, (batch_size, self.output_dim), 'prediction')
        
        return prediction, state