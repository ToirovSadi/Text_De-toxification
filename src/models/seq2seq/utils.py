import torch

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)

# preprocess the raw text to pass to the model
def preprocess_text(text: str, max_sent_size: int, vocab):
    text = ['<sos>'] + word_tokenize(text.lower()) + ['<eos>']
    if len(text) > max_sent_size:
        raise ValueError(f'text is too large for the model max_sent_size: {max_sent_size}, but got {len(text)}. Use utils.predict_text it can handle large text')

    tokens = vocab.lookup_indices(text)
    while len(tokens) < max_sent_size:
        tokens.append(vocab['<pad>'])

    return torch.tensor(tokens).reshape(max_sent_size, 1)


# postprocess the text
def postprocess_text(text: list[str], detokenize=True) -> str:
    # TODO: do proper detokenization
    if detokenize:
        return TreebankWordDetokenizer().detokenize(text)
    return ' '.join(text)
