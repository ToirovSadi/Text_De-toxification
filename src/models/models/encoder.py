import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        num_layers=1,
        dropout=0,
        padding_idx=None,
    ):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, *args):
        # x.shape: (num_steps, batch_size)
        embs = self.dropout(self.embedding(x))
        
        # embs.shape: (num_steps, batch_size, embed_dim)
        outputs, state = self.rnn(embs)
        # outputs.shape: (num_steps, batch_size, hidden_dim)
        # state.shape: (num_layers, batch_size, hidden_dim)
        
        return outputs, state