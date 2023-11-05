import torch.nn as nn

class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab = vocab
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, hidden, context):
        # x.shape: [batch_size]
        x = x.unsqueeze(1) # -> x.shape: [batch_size, 1]
        
        emb = self.dropout(self.embedding(x))
        # emb.shape: [batch_size, 1, embed_dim]
        output, hidden = self.rnn(emb, hidden)
        # outputs.shape: [batch_size, 1, hidden_dim]
        # hidden.shape: [num_layers, batch_size, hidden_dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction.shape: [batch_size, output_dim]
        
        return prediction, hidden
