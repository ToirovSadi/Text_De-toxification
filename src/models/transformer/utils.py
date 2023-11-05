import torch

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)
from queue import PriorityQueue

from src.data.preprocessing import preprocess_text as prep

# preprocess the raw text to pass to the model
def preprocess_text(text: str, max_sent_size: int, vocab):
    text = ['<sos>'] + prep(text) + ['<eos>']
    if len(text) > max_sent_size:
        raise ValueError(f'text is too large for the model max_sent_size: {max_sent_size}, but got {len(text)}. Use utils.predict_text it can handle large text')

    tokens = vocab.lookup_indices(text)
    while len(tokens) < max_sent_size:
        tokens.append(vocab['<pad>'])

    return torch.tensor(tokens).reshape(1, max_sent_size)


# postprocess the text
def postprocess_text(text: list[str], detokenize=True) -> str:
    # remove the specials
    specials = ['<pad>', '<unk>', '<sos>', '<eos>']
    text = [x for x in text if x not in specials]
    
    # TODO: do proper detokenization
    if detokenize:
        return TreebankWordDetokenizer().detokenize(text)
    return ' '.join(text)

### Prediction Functions For Transformer ###
# perform greed search to predict
def greedy_search(model, src, return_attention=False):
    # src.shape: [batch_size, num_steps]
    batch_size = src.shape[0]
    if batch_size != 1:
        raise ValueError("it should be one sentence at a time, batch_size == 1")

    # we will use decoder vocab to get tokens
    vocab = model.decoder.vocab        
    max_sent_size = model.max_sent_size
    outputs = []
    model.eval()
    
    with torch.no_grad():
        # pass the src through the encoder
        src_mask = model.mask_src(src)
        enc_outputs = model.encoder(src, src_mask)
    
    eos_idx = model.decoder.vocab['<eos>']
    res = [model.decoder.vocab['<sos>']]
    with torch.no_grad():
        for t in range(1, max_sent_size):
            trg = torch.LongTensor(res).unsqueeze(0).to(model.device)
            
            trg_mask = model.mask_trg(trg)
            pred = model.decoder(trg, enc_outputs, src_mask, trg_mask, return_attention=return_attention)
            if return_attention:
                pred, attention = pred
            # pred.shape: [batch_size, max_sent_size, output_dim]
            
            top1 = pred[:, -1, :].argmax(1)
            res.append(top1.item())
            if top1 == eos_idx:
                break

    if return_attention:
        res = (res, attention)
    
    return res


## Beam Search Implementation ##
# beam search node, to reconstruct the output
class BeamNode:
    def __init__(self, dec_in, log_prob, length):
        self.dec_in = dec_in
        self.log_prob = log_prob
        self.length = length
        
    def eval(self, alpha=0.75):
        # alpha: more penalty for longer sentences
        return self.log_prob / (self.length ** alpha)
    
    def __ln__(self, other):
        return self.log_prob < other.log_prob
    
# perform beam search to predict
# note beam search does not return attention weight
# you have to use greedy_search with return_attention=True
# maybe added in future
def beam_search(model, src, beam_width=5, num_candidates=3, max_steps=2000, max_candidate_coef=3):
    # if beam search by it self working slowly try to reduce `max_steps` or `max_candidate_coef`
    max_candidates = num_candidates * max_candidate_coef
    
    # init state
    q = PriorityQueue()
    qsize = 0
    candidates = []
    eos_idx = model.decoder.vocab['<eos>']
    sos_idx = model.decoder.vocab['<sos>']
    batch_size = src.shape[0]
    if batch_size != 1:
        raise ValueError("it should be one sentence at a time, batch_size == 1")
    
    # first steps
    with torch.no_grad():
        src_mask = model.mask_src(src)
        enc_outputs = model.encoder(src, src_mask)
    dec_in = torch.LongTensor([[sos_idx]]).to(model.device)
    node = BeamNode(dec_in, log_prob=0, length=1)
    q.put((-node.eval(), node))
    # start the beam search
    with torch.no_grad():
        while q.qsize() > 0 and qsize <= max_steps:
            score, node = q.get()
            # print(node.dec_in[:, -1])
            if node.dec_in[:, -1] == eos_idx and node.length > 1:
                candidates.append((score, node))
                if len(candidates) > max_candidates:
                    break
                continue
            if node.length > model.max_sent_size:
                continue
            
            trg_mask = model.mask_trg(node.dec_in)
            preds = model.decoder(node.dec_in, enc_outputs, src_mask, trg_mask, return_attention=False)
            # print("preds.shape:", preds.shape)
            preds = torch.log_softmax(preds[:, -1, :], dim=1)
            # print("preds.shape:", preds.shape)
            # take only top beam_width words
            topk, indices = torch.topk(preds, beam_width)
            # topk.shape: [1, beam_width]
            # print("topk.shape:", topk.shape)
            for i in range(beam_width):
                prob = topk[0][i].item()
                dec_in = indices[0][i].view(1, -1)
                dec_in = torch.cat((node.dec_in, dec_in), dim=1)
                next_node = BeamNode(
                    dec_in,
                    log_prob=node.log_prob + prob,
                    length=node.length + 1,
                )
                q.put((-next_node.eval(), next_node))
                
            qsize += beam_width - 1
    
    # if not enough answer, take from top of queue
    while len(candidates) < max_candidates and q.qsize() > 0:
        candidates.append(q.get())
        
    # sort candidates, and take first `num_candidates`
    candidates = sorted(candidates, key=lambda x: x[0])[:num_candidates]
    
    outputs = []
    for score, node in candidates:
        outputs.append(node.dec_in.cpu().detach().numpy()[0])
        
    return outputs