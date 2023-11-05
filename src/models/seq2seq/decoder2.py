import torch.nn as nn
import torch

class Decoder2(nn.Module):
    def __init__(
        self,
        output_dim,
        embed_dim,
        hidden_dim,
        num_layers=1,
        dropout=0,
        vocab=None,
        padding_idx=None,
    ):
        super(Decoder2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.GRU(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, output_dim),
        ) 
        self.dropout = nn.Dropout(dropout)
        self.vocab = vocab
        
    def forward(self, x, hidden, context):
        # x.shape: [batch_size]
        # hidden.shape: [n_layers, batch_size, hidden_dim]
        # context: [n_layers, batch_size, hidden_dim]
        x = x.unsqueeze(1) # x.shape: [batch_size, 1]
        emb = self.dropout(self.embedding(x))
        # emd.shape: [batch_size, 1, embed_dim]
        
        emb = torch.cat((emb, context), dim=2)
        # emd.shape: [batch_size, 1, hidden_dim + embed_dim]
        outputs, hidden = self.rnn(emb, hidden)
        # outputs.shape: [batch_size, 1, hidden_dim]
        # hidden.shape: [n_layers, batch_size, hidden_dim]
        
        output = torch.cat((emb, outputs), dim=2)
        # output.shape: [batch_size, 1, 2 * hidden_dim + embed_dim]
        output = self.fc_out(output.squeeze(1))
        # output.shape: [batch_size, output_dim]
        
        return output, hidden
        