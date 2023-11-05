import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % heads == 0
        
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.heads_dim = hidden_dim // heads
        self.scale = torch.sqrt(torch.tensor(self.heads_dim)).to(device)
        
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        # (key, value, query).shape: [batch_size, max_sent_size, hidden_dim]
        
        batch_size, max_sent_size, hidden_dim = key.shape
            
        key = self.fc_k(key).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        value = self.fc_k(value).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        query = self.fc_k(query).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        # key.shape: [batch_size, heads, max_sent_size, heads_dim]
        
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        # energy.shape: [batch_size, heads, max_sent_size, max_sent_size]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), value).permute(0, 2, 1, 3).contiguous()
        # x.shape: [batch_size, max_sent_size, heads, heads_dim]
        x = self.fc_o(x.view(batch_size, -1, self.hidden_dim))
        if return_attention:
            return x, attention
        return x