import torch.nn as nn
import torch

from .attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, heads, ff_expantion, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(
            hidden_dim,
            heads,
            dropout,
            device,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.encoder_attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expantion),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * ff_expantion, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, enc_out, src_mask=None, trg_mask=None, return_attention=False):
        # x.shape: [batch_size, max_sent_size]
        # enc_out.shape: [batch_size, max_sent_size, hidden_dim]
        # src_mask: [batch_size, max_sent_size]
        # trg_mask: [batch_size, max_sent_size]
        
        _x = self.dropout(self.self_attention(x, x, x, trg_mask))
        # _x.shape: [batch_size, max_sent_size, hidden_dim]
        
        x = self.norm1(_x + x)
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        
        _x, attention = self.encoder_attention(x, enc_out, enc_out, src_mask, return_attention=True)
        
        x = self.norm2(self.dropout(_x) + x)
        
        x = self.norm3(self.dropout(self.ff(x)) + x)
        
        if return_attention:
            return x, attention
        return x
        

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, heads, ff_expantion, dropout, device, max_size, vocab):
        super(Decoder, self).__init__()
        self.padding_idx = vocab['<pad>']
        self.token_embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=self.padding_idx)
        self.pos_embedding = nn.Embedding(max_size, hidden_dim)

        
        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden_dim=hidden_dim,
                heads=heads,
                ff_expantion=ff_expantion,
                dropout=dropout,
                device=device,
            ) for _ in range(num_layers)])
        
        self.fc_out = nn.LazyLinear(output_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim)).to(device)
        self.device = device
        self.vocab = vocab
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask, return_attention=False):
        # x.shape: [batch_size, max_sent_size]
        # src_mask.shape: [batch_size, max_sent_size]
        # trg_mask.shape: [batch_size, max_sent_size]
        batch_size, max_sent_size = x.shape
        
        emb = self.token_embedding(x)
        # emb.shape: [batch_size, max_sent_size, hidden_dim]
        
        pos = torch.arange(0, max_sent_size).reshape(1, max_sent_size).repeat(batch_size, 1).to(self.device)
        pos = self.pos_embedding(pos)
        x = self.dropout(emb * self.scale + pos)
        
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask, return_attention=return_attention)
            if return_attention:
                x, attention = x
        
        x = self.fc_out(x)
        
        if return_attention:
            return x, attention
        
        return x
        