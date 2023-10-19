import torch

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')

# preprocess the raw text to pass to the model
def preprocess_text(text: str, max_sent_size: int, vocab):
    text = word_tokenize(text.lower())
    if len(text) > max_sent_size:
        raise ValueError(f'text is too large for the model max_sent_size: {max_sent_size}, but got {len(text)}. Use utils.predict_text it can handle large text')

    tokens = vocab.lookup_indices(text)
    while len(tokens) < max_sent_size:
        tokens.append(vocab['<pad>'])

    return torch.tensor(tokens).reshape(max_sent_size, 1)


# postprocess the text
def postprocess_text(text: list[str], detokenize) -> str:
    # TODO: do proper detokenization
    if detokenize:
        return TreebankWordDetokenizer().detokenize(text)
    return ' '.join(text)


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