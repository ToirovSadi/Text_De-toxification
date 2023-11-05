import torch.nn as nn
import torch

from .utils import preprocess_text, postprocess_text
from .utils import greedy_search, beam_search

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, max_sent_size, device):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_sent_size = max_sent_size
    
    def mask_src(self, x):
        # x.shape: [batch_size, max_sent_size]
        x = x.unsqueeze(1).unsqueeze(2)
        return (x != self.encoder.padding_idx)
    
    def mask_trg(self, x):
        # x.shape: [batch_size, max_sent_size]
        trg_pad_mask = (x != self.decoder.padding_idx).unsqueeze(1).unsqueeze(2)
        max_sent_size = x.shape[1]
        mask = torch.tril(torch.ones((max_sent_size, max_sent_size), device=self.device)).bool()
        return trg_pad_mask & mask
    
    def forward(self, src, trg, t=None, return_attention=False):
        # remove <eos> from trg
        trg = trg[:, :-1]
        
        src_mask = self.mask_src(src)
        trg_mask = self.mask_trg(trg)
        enc_output = self.encoder(src, src_mask)
        outputs = self.decoder(trg, enc_output, src_mask, trg_mask, return_attention=return_attention)
        return outputs
    
    # by default it uses greedy search
    def predict(self, src, post_process_text=True, use_beam_search=False, num_candidates=1, beam_width=5, return_attention=False):
        if type(src) is str:
            src = preprocess_text(src, max_sent_size=self.max_sent_size, vocab=self.encoder.vocab)
        src = src.to(self.device)
        batch_size, num_steps = src.shape
        
        if batch_size != 1:
            raise ValueError("batch_size should be one, i.e. one sentence at a time")
        
        if use_beam_search:
            sentences = beam_search(
                self,
                src,
                beam_width=beam_width,
                num_candidates=num_candidates
            )
            res = []
            for sent in sentences:
                sent = self.decoder.vocab.lookup_tokens(sent)
                if post_process_text:
                    sent = postprocess_text(sent)

                res.append(sent)
        else:
            res = greedy_search(self, src, return_attention=return_attention)
            if return_attention:
                res, attention = res
            res = self.decoder.vocab.lookup_tokens(res)
            if post_process_text:
                res = postprocess_text(res)
            
            if return_attention:
                res = (res, attention)
        
        return res