import torch.nn as nn
import torch
from random import random

from .utils import check_shape
from .utils import preprocess_text, postprocess_text

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device, max_sent_size, vocab):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_size = max_sent_size
        self.vocab = vocab
        
    def forward(self, src, trg, teacher_force_ratio=0.5):
        # src.shape: [num_steps, batch_size]
        # trg.shape: [num_steps, batch_size]
        
        num_steps, batch_size = src.shape
        output_dim = self.decoder.output_dim
        outputs = torch.zeros((num_steps, batch_size, output_dim), device=self.device)
        
        enc_outputs, hidden = self.encoder(src)
        # enc_outputs.shape: [num_steps, batch_size, enc_hidden_dim * 2]
        # hidden.shape: [batch_size, dec_hidden_dim]
        check_shape(enc_outputs, (num_steps, batch_size, self.encoder.hidden_dim * 2), 'enc_outputs')
        check_shape(hidden, (batch_size, self.decoder.hidden_dim), 'hidden')
        
        dec_input = trg[0, :] # <sos>
        for t in range(1, num_steps):
            pred, hidden = self.decoder(dec_input, hidden, enc_outputs)
            # pred.shape: [batch_size, output_dim]
            # hidden.shape: [batch_size, dec_hidden_dim]
            
            outputs[t] = pred
            
            top1 = pred.argmax(1)
            
            teacher_force = random() < teacher_force_ratio
            
            dec_input = trg[t, :] if teacher_force else top1
        
        return outputs
    
    
    def predict(self, src, post_process_text=True, **args) -> list[str]:
        if type(src) is str:
            src = preprocess_text(src, max_sent_size=self.max_sent_size, vocab=self.vocab)
        src = src.to(self.device)
        num_steps, batch_size = src.shape
        
        if batch_size != 1:
            raise ValueError("batch_size should be one, i.e. one sentence at a time")
        
        res = []
        self.eval()
        eos_idx = self.vocab['<eos>']
        with torch.no_grad():
            # pass the src through the encoder
            outputs_enc, state_dec = self.encoder(src)
            input_dec = src[0, :] # <sos>
            
            for t in range(1, num_steps):
                pred, state_dec = self.decoder(input_dec, state_dec, outputs_enc)
                # pred.shape: [batch_size, output_dim]

                top1 = pred.argmax(1)
                if top1 == eos_idx:
                    break
                res.append(self.vocab.lookup_tokens(top1.cpu().detach().numpy())[0])
                input_dec = top1
        
        if post_process_text:
            return postprocess_text(res)
        
        return res