import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        num_layers=1,
        dropout=0,
        vocab=None,
        padding_idx=None,
    ):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab = vocab
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        # x.shape: (batch_size, num_steps)
        embs = self.dropout(self.embedding(x))
        
        # embs.shape: (batch_size, num_steps, embed_dim)
        outputs, state = self.rnn(embs)
        # outputs.shape: (batch_size, num_steps, hidden_dim)
        # state[0].shape: (num_layers, num_steps, hidden_dim)
        
        return outputs, state