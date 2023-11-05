import contractions
from spellchecker import SpellChecker
import regex

import nltk
from nltk import word_tokenize
nltk.download('punkt', quiet=True)

spell = SpellChecker(distance=1)

def correct_word(word, unk=None):
    if unk is None:
        unk = word
    corrected = spell.correction(word)
    return unk if corrected is None else corrected

def correct_spellings(text: list[str]) -> list[str]:
    corrected_text = []
    misspelled_words = spell.unknown(text)
    for word in text:
        if word in misspelled_words:
            corrected_text.extend([correct_word(x) for x in regex.findall('[a-zA-Z0-9_]+', word)])
        else:
            corrected_text.append(word)
    return corrected_text

def convert_contractions(text: list[str]) -> list[str]:
    return [contractions.fix(word) for word in text]

def preprocess_text(text: str) -> list[str]:
    # Step 1: lower the text
    text = text.lower()
    
    # Step 2: Convert Contractions to normal form
    text = contractions.fix(text)
    
    # Step 3: Word Tokenization
    text = word_tokenize(text)
    
    # Step 4: Correct Spelling
    text = correct_spellings(text)
    
    # Step 5: Remove constractions that appeard after spell correcting
    text = convert_contractions(text)
    
    return text