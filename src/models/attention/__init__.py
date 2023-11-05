import torch.nn as nn
import torch
from random import random

from .utils import check_shape
from .utils import preprocess_text, postprocess_text
from .utils import greedy_search, beam_search

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device, max_sent_size):
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_size = max_sent_size
        
    def forward(self, src, trg, teacher_force_ratio=0.5):
        # src.shape: [batch_size, num_steps]
        # trg.shape: [batch_size, num_steps]
        
        batch_size, num_steps = trg.shape
        
        # outputs_dec.shape: [batch_size, num_steps-1, decoder.output_dim]
        outputs = []
        dec_input = trg[:, 0] # <sos>
        
        enc_outputs, hidden = self.encoder(src)
        for t in range(1, self.max_sent_size):
            pred, hidden = self.decoder(dec_input, hidden, enc_outputs)
            # pred.shape: [batch_size, output_dim]
            # hidden.shape: [batch_size, dec_hidden_dim]
            outputs.append(pred)
            
            top1 = pred.argmax(1)
            
            teacher_force = random() < teacher_force_ratio
            
            dec_input = trg[:, t] if teacher_force else top1
        
        return torch.stack(outputs, dim=1)
    
    
    def predict(
        self,
        src,
        beam=False,
        beam_search_width=3,
        beam_search_num_candidates=3,
        post_process_text=True,
    ) -> list[str]:
        if type(src) is str:
            src = preprocess_text(src, max_sent_size=self.max_sent_size, vocab=self.encoder.vocab)
        src = src.to(self.device)
        batch_size, num_steps = src.shape
        
        if batch_size != 1:
            raise ValueError("batch_size should be one, i.e. one sentence at a time")
        
        if beam:
            sentences = beam_search(
                self,
                src,
                beam_width=beam_search_width,
                num_candidates=beam_search_num_candidates,
            )
            res = []
            for sent in sentences:
                sent = self.decoder.vocab.lookup_tokens(sent)
                if post_process_text:
                    sent = postprocess_text(sent)

                res.append(sent)
        else:
            res = greedy_search(self, src)
            res = self.decoder.vocab.lookup_tokens(res)
            if post_process_text:
                res = postprocess_text(res)
        
        return res