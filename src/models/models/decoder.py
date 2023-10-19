import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embed_dim,
        hidden_dim,
        num_layers=1,
        dropout=0,
        padding_idx=None,
    ):
        super(Decoder, self).__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, hidden):
        # x.shape: [batch_size]
        x.unsqueeze_(0) # -> x.shape: [1, batch_size]
        
        emb = self.dropout(self.embedding(x))
        # emb.shape: [1, batch_size, embed_dim]
        
        output, hidden = self.rnn(emb, hidden)
        # outputs.shape: [1, batch_size, num_hidden]
        # hidden.shape: [num_layers, batch_size, num_hidden]
        
        prediction = self.fc(output.squeeze(0))
        # prediction.shape: [batch_size, output_dim]
        
        return prediction, hidden
