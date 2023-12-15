import torch

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from queue import PriorityQueue

# preprocess the raw text to pass to the model
def preprocess_text(text: str, max_sent_size: int, vocab):
    text = ['<sos>'] + word_tokenize(text.lower()) + ['<eos>']
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

def calculate_bleu(dataset, model, **args):
    preds = []
    trgs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            toxic_sent, neutral_sent = dataset[i]
            toxic_sent = toxic_sent.to(model.device).unsqueeze(0)
            pred = model.predict(toxic_sent, post_process_text=False, **args)
            
            pred = pred[1:-1] # remove <sos> and <eos>
            
            neutral_sent = model.decoder.vocab.lookup_tokens(neutral_sent.numpy())
            neutral_sent = neutral_sent[1:] # remove <sos>
            neutral_sent = neutral_sent[:neutral_sent.index('<eos>')]
            
            preds.append(pred)
            trgs.append([neutral_sent])
        
    return bleu_score(preds, trgs)

# # perform greed search to predict
# def greedy_search(model, src, use_encoder_out=False):
#     # src.shape: [batch_size, num_steps]
#     batch_size = src.shape[0]
#     if batch_size != 1:
#         raise ValueError("it should be one sentence at a time, batch_size == 1")
    
#     # we will use decoder vocab to get tokens
#     vocab = model.decoder.vocab        
#     max_sent_size = model.max_sent_size
#     outputs = []
    
#     # pass input through the encoder
#     with torch.no_grad():
#         encoder_out, hidden = model.encoder(src)

#     eos_idx = vocab['<eos>']
#     if use_encoder_out:
#         context = encoder_out
#     else:
#         context = hidden.swapaxes(0, 1).clone()
#     dec_in = torch.empty(batch_size, device=model.device, dtype=torch.long).fill_(vocab['<sos>'])
#     with torch.no_grad():
#         for i in range(max_sent_size):
#             preds, hidden = model.decoder(dec_in, hidden, context)
#             top1 = preds.argmax(1)
#             if top1 == eos_idx:
#                 break
#             outputs.append(top1.cpu().detach().numpy()[0])
#             dec_in = top1

#     return outputs


# ### Beam Search Implementation
# # beam search node, to reconstruct the output
# class BeamNode:
#     def __init__(self, hidden, dec_in, log_prob, length, prev_node=None):
#         self.hidden = hidden
#         self.dec_in = dec_in
#         self.log_prob = log_prob
#         self.length = length
#         self.prev_node = prev_node
        
#     def eval(self, alpha=0.75):
#         # alpha: more penalty for longer sentences
#         return self.log_prob / (self.length ** alpha)
    
#     def __ln__(self, other):
#         return self.log_prob < other.log_prob
    
# # perform beam search to predict
# def beam_search(model, src, beam_width=5, num_candidates=3, max_steps=2000, max_candidate_coef=3, use_encoder_out=False):
#     # if beam search by it self working slowly try to reduce `max_steps` or `max_candidate_coef`
#     max_candidates = num_candidates * max_candidate_coef
    
#     # init state
#     q = PriorityQueue()
#     qsize = 0
#     candidates = []
#     eos_idx = model.decoder.vocab['<eos>']
#     sos_idx = model.decoder.vocab['<sos>']
#     batch_size = src.shape[0]
#     if batch_size != 1:
#         raise ValueError("it should be one sentence at a time, batch_size == 1")
    
#     # first steps
#     with torch.no_grad():
#         encoder_out, hidden = model.encoder(src)
#     if use_encoder_out:
#         context = encoder_out
#     else:
#         context = hidden.swapaxes(0, 1).clone()
#     dec_in = torch.LongTensor([sos_idx]).to(model.device)
#     node = BeamNode(hidden, dec_in, log_prob=0, length=1)
#     q.put((-node.eval(), node))
#     # start the beam search
#     with torch.no_grad():
#         while q.qsize() > 0 and qsize <= max_steps:
#             score, node = q.get()

#             if node.dec_in == eos_idx and node.prev_node != None:
#                 candidates.append((score, node))
#                 if len(candidates) > max_candidates:
#                     break
#                 continue
#             if node.length > model.max_sent_size:
#                 continue
        
#             preds, hidden = model.decoder(node.dec_in, node.hidden, context)
#             preds = torch.log_softmax(preds, dim=1)
            
#             # take only top beam_width words
#             topk, indices = torch.topk(preds, beam_width)
#             # topk.shape: [1, beam_width]
            
#             for i in range(beam_width):
#                 prob = topk[0][i].item()
#                 dec_in = indices[0][i].view(-1)
                
#                 next_node = BeamNode(
#                     hidden,
#                     dec_in,
#                     log_prob=node.log_prob + prob,
#                     length=node.length + 1,
#                     prev_node=node,
#                 )
#                 q.put((-next_node.eval(), next_node))
                
#             qsize += beam_width - 1
    
#     # if not enough answer, take from top of queue
#     while len(candidates) < max_candidates and q.qsize() > 0:
#         candidates.append(q.get())
        
#     # sort candidates, and take first `num_candidates`
#     candidates = sorted(candidates, key=lambda x: x[0])[:num_candidates]
    
#     outputs = []
#     for score, node in candidates:
#         ans = []
#         while node.prev_node != None:
#             ans.append(node.dec_in.item())
#             node = node.prev_node
        
#         # reverse the result
#         outputs.append(ans[::-1])
        
#     return outputs

# to predict the large text (if it's more than MAX_SENT_SIZE)
def predict_text(text: str, model, post_process_text=True):
    max_sent_size = model.max_sent_size
    vocab = model.vocab
    res = []
    for text in sent_tokenize(text):
        text = word_tokenize(text.lower())
        first_batch = ['<sos>'] + text[:max_sent_size - 2] + ['<eos>']
        tokens = vocab.lookup_indices(first_batch)
        while len(tokens) < max_sent_size:
            tokens.append(vocab['<pad>'])
        tokens = torch.tensor(tokens).reshape(max_sent_size, 1)
        ans = model.predict(tokens, post_process_text=False)
        # next take one more token and make prediction add last one
        for i in range(1, len(text) - max_sent_size + 3):
            batch = ['<sos>'] + text[i:i + max_sent_size - 2] + ['<eos>']
            tokens = vocab.lookup_indices(batch)
            while len(tokens) < max_sent_size:
                tokens.append(vocab['<pad>'])
            tokens = torch.tensor(tokens).reshape(max_sent_size, 1)
            ans.append(model.predict(tokens, post_process_text=False)[-1])
        if post_process_text:
            ans = postprocess_text(ans)
        res.append(ans)
    
    if post_process_text:
        res = postprocess_text(res)
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)