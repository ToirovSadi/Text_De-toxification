import torch.nn as nn
import torch
from random import random
import os

from .encoder import Encoder
from .decoder import Decoder

from .utils import preprocess_text, postprocess_text

class EncoderDecoder(nn.Module):
    def __init__(self, encoder=None, decoder=None, device=None, max_sent_size=None, vocab=None):
        super(EncoderDecoder, self).__init__()
        
        if encoder is None or decoder is None:
            return
        
        assert encoder.num_layers == decoder.num_layers, "num_layers of encoder and decoder should be the same,"\
        f"but got encoder.num_layers: {encoder.num_layers} and decoder.num_layers: {decoder.num_layers}"
        
        assert encoder.hidden_dim == decoder.hidden_dim, "hidden_dim of encoder and decoder should be the same,"\
        f"but got encoder.hidden_num: {encoder.hidden_dim} and decoder.hidden_dim: {decoder.hidden_dim}"
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_size = max_sent_size
        self.vocab = vocab
    
    def predict(self, src, post_process_text=True, **args) -> list[str]:
        if type(src) is str:
            src = preprocess_text(src, max_sent_size=self.max_sent_size, vocab=self.vocab)
        src = src.to(self.device)
        num_steps, batch_size = src.shape
        
        if batch_size != 1:
            raise ValueError("batch_size should be one, i.e. one sentence at a time")
        
        res = []
        eos_idx = self.vocab['<eos>']
        with torch.no_grad():
            # pass the src through the encoder
            outputs_enc, state_dec = self.encoder(src)
            input_dec = src[0, :] # <sos>
            
            for t in range(1, num_steps):
                pred, state_dec = self.decoder(input_dec.clone(), state_dec)
                # pred.shape: [batch_size, output_dim]

                top1 = pred.argmax(1)
                if top1 == eos_idx:
                    break
                res.append(self.vocab.lookup_tokens(top1.cpu().detach().numpy())[0])
                input_dec = top1
        
        if post_process_text:
            return postprocess_text(res)
        
        return res
