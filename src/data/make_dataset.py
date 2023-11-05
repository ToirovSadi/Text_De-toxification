import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator

class ParanmtDataset(Dataset):
    def __init__(
        self,
        path: str,
        max_sent_size,
        vocabs=None,
        train: bool=True,
        split_ratio=(0.9, 0.1),
        take_first=None,
        seed=None,
    ):
        self.train = train
        self.max_sent_size = max_sent_size
        
        if vocabs is not None: # if not given, please build it with build_vocab method 
            self.toxic_vocab = vocabs[0]
            self.neutral_vocab = vocabs[1]
        
        if type(split_ratio) not in [list, tuple]:
            split_ratio = [split_ratio]
        
        df = pd.read_csv(path, sep='\t', index_col=0)
        
        self.dataframe = df.copy()
        
        # let's convert columns toxic_sent and neutral_sent as a list
        df.loc[:, 'toxic_sent'] = df['toxic_sent'].apply(lambda x: x.split(' '))
        df.loc[:, 'neutral_sent'] = df['neutral_sent'].apply(lambda x: x.split(' '))
        
        # drop the column with sentence size more than MAX_SENT_SIZE including <sos>, <eos>
        df = df[df['toxic_sent'].apply(len) <= self.max_sent_size-2]
        df = df[df['neutral_sent'].apply(len) <= self.max_sent_size-2]
        
        # split into train and val
        train_idx, val_idx = train_test_split(list(df.index), train_size=split_ratio[0], random_state=seed)
        if train:
            self.df = df.loc[train_idx].reset_index(drop=True)
        else:
            self.df = df.loc[val_idx].reset_index(drop=True)
            
        if take_first != None:
            self.df = self.df[:take_first]
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        toxic_sent = self.df.loc[idx, 'toxic_sent']
        neutral_sent = self.df.loc[idx, 'neutral_sent']
        
        toxic_sent = self._preprocess_text(toxic_sent, self.toxic_vocab)
        neutral_sent = self._preprocess_text(neutral_sent, self.neutral_vocab)
        
        return toxic_sent, neutral_sent
        
    
    def build_vocab(self, df=None, min_freq=1, specials=None, max_tokens=None):
        if df is None:
            df = self.df

        def yield_tokens(df, column):
            for i in df.index:
                sent = df[column][i]
                yield sent
        
        # build vocab for toxic sentence
        self.toxic_vocab = build_vocab_from_iterator(
            yield_tokens(df, 'toxic_sent'),
            min_freq=min_freq,
            specials=specials,
            max_tokens=max_tokens,
        )
        self.toxic_vocab.set_default_index(specials.index('<unk>'))
        
        # build vocab for neutral sentence
        self.neutral_vocab = build_vocab_from_iterator(
            yield_tokens(df, 'neutral_sent'),
            min_freq=min_freq,
            specials=specials,
            max_tokens=max_tokens,
        )
        self.neutral_vocab.set_default_index(specials.index('<unk>'))
    
    def _preprocess_text(self, text, vocab):
        assert vocab is not None, "vocab is not defined, please build vocab or pass it as argument"
        
        text = ['<sos>'] + text[:self.max_sent_size-2] + ['<eos>']
        text = vocab.lookup_indices(text)
        while len(text) < self.max_sent_size:
            text.append(vocab['<pad>'])
        
        return torch.tensor(text)
    
    
def collate_batch(batch): # collate_batch to make seq_len first
    toxic_sent = []
    neutral_sent = []
    for _toxic_sent, _neutral_sent in batch:
        toxic_sent.append(_toxic_sent)
        neutral_sent.append(_neutral_sent)
    return torch.stack(toxic_sent, dim=1), torch.stack(neutral_sent, dim=1)