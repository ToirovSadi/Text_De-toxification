import torch.nn as nn
import torch
from random import random
from copy import deepcopy

from .seq2seq import EncoderDecoder

class Seq2Seq(EncoderDecoder):
    def forward(self, src, trg, teacher_forcing_ration=0.5):
        # src.shape: [num_steps, batch_size]
        # trg.shape: [num_steps, batch_size]
        num_steps, batch_size = src.shape
        
        # outputs_dec.shape: [num_steps, batch_size, decoder.output_dim]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(num_steps, batch_size, output_dim, device=self.device)
        
        # pass the src through the encoder
        _, hidden = self.encoder(src)
        
        input_dec = trg[0, :] # <sos>
        for t in range(1, num_steps):
            pred, hidden = self.decoder(input_dec, hidden)
            # pred.shape: [batch_size, output_dim]
            
            outputs[t] = pred
            
            top1 = pred.argmax(1)
            
            teacher_force = random() < teacher_forcing_ration
            
            input_dec = trg[t].clone() if teacher_force else top1
        
        # outputs.shape: [num_steps, batch_size, output_dim]
        return outputs
