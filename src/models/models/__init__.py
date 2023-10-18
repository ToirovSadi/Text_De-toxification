import torch.nn as nn
import torch
from random import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        assert encoder.num_layers == decoder.num_layers, "num_layers of encoder and decoder should be the same,"\
        f"but got encoder.num_layers: {encoder.num_layers} and decoder.num_layers: {decoder.num_layers}"
        
        assert encoder.hidden_dim == decoder.hidden_dim, "hidden_dim of encoder and decoder should be the same,"\
        f"but got encoder.hidden_num: {encoder.hidden_dim} and decoder.hidden_dim: {decoder.hidden_dim}"
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    
    def forward(self, src, trg, teacher_forcing_ration=0.5):
        # src.shape: [num_steps, batch_size]
        # trg.shape: [num_steps, batch_size]
        num_steps, batch_size = src.shape
        
        # pass the src through the encoder
        outputs_enc, state_enc = self.encoder(src)
        
        # outputs_dec.shape: [num_steps, batch_size, decoder.output_dim]
        output_dim = self.decoder.output_dim
        outputs_dec = torch.zeros(num_steps, batch_size, output_dim).to(self.device)
        
        input_dec = trg[0, :] # <sos>
        
        state_dec = state_enc # pass to the first decoder the context vector
        for t in range(1, num_steps):
            pred, state_dec = self.decoder(input_dec.clone(), state_dec)
            # pred.shape: [batch_size, output_dim]
            
            outputs_dec[t] = pred
            
            top1 = pred.argmax(1)
            
            teacher_force = random() < teacher_forcing_ration
            
            input_dec = trg[t] if teacher_force else top1
        
        # outputs.shape: [num_steps, batch_size, output_dim]
        return outputs_dec
    
    def preprocess_text(text: str, max_sent_size: int, vocab) -> list[str]:
        from nltk import word_tokenize
        # preprocess the text
        text = word_tokenize(text.lower())

        tokens = vocab.lookup_indices(text)
        while len(tokens) < max_sent_size:
            tokens.append(vocab['<pad>'])
        
        return torch.tensor(tokens).reshape(max_sent_size, 1)
    
    def predict(src) -> list[str]:
        src = src.to(self.device)
        num_steps, batch_size = src.shape
        
        if batch_size != 1:
            raise ValueError("batch_size should be one, i.e. one sentence at a time")
        
        # pass the src through the encoder
        outputs_enc, state_dec = self.encoder(src)
        input_dec = src[0, :] # <sos>
        res = []
        for t in range(1, num_steps):
            pred, state_dec = self.decoder(input_dec.clone(), state_dec)
            # pred.shape: [batch_size, output_dim]

            top1 = pred.argmax(1)
            if top1 == vocab['<eos>']:
                break
            res.append(vocab.lookup_tokens(top1.cpu().detach().numpy())[0])
            input_dec = top1
        
        return res