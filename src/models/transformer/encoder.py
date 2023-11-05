import torch.nn as nn
import torch

from .attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, heads, ff_expantion, dropout, device):
        super(EncoderBlock, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expantion),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * ff_expantion, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        
        _x = self.dropout(self.attention(x, x, x, mask))
        # _x.shape: [batch_size, max_sent_size, hidden_dim]
        
        x = self.norm1(_x + x)
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        
        _x = self.dropout(self.ff(x))
        # _x.shape: [batch_size, max_sent_size, hidden_dim]
        
        x = self.norm2(_x + x)
        
        # _x.shape: [batch_size, max_sent_size, hidden_dim]
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, heads, ff_expantion, dropout, device, max_size, vocab):
        super(Encoder, self).__init__()
        
        self.padding_idx = vocab['<pad>']
        self.token_embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=self.padding_idx)
        self.pos_embedding = nn.Embedding(max_size, hidden_dim)
        
        self.layers = nn.ModuleList([
            EncoderBlock(
                hidden_dim=hidden_dim,
                heads=heads,
                ff_expantion=ff_expantion,
                dropout=dropout,
                device=device,
            ) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(hidden_dim)).to(device)
        self.device = device
        self.vocab = vocab
        
    def forward(self, x, mask):
        # x.shape: [batch_size, max_sent_size]
        # mask.shape: [batch_size, max_sent_size]
        batch_size, max_sent_size = x.shape
        emb = self.token_embedding(x) * self.scale
        # emb.shape: [batch_size, max_sent_size, hidden_dim]
        pos = torch.arange(0, max_sent_size).reshape(1, max_sent_size).repeat(batch_size, 1).to(self.device)
        # pos.shape: [batch_size, max_sent_size]
        
        pos = self.pos_embedding(pos)
        # pos.shape: [batch_size, max_sent_size, hidden_dim]
        x = self.dropout(emb + pos)
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        
        for layer in self.layers:
            x = layer(x, mask)
        
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        return x