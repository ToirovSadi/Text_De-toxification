# this is Streamlit app, for you to test the models

import streamlit as st
import torch
import gdown
import json
import os

st.title('Text De-toxification')
st.subheader('This is a demo for text de-toxification.\nSelect a model and input a sentence to test it.')

@st.cache_data(ttl=24*3600, max_entries=1)
def load_model(model_name):
    if model_name == 't5-small (max_len=128)':
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('ToirovSadi/t5-small')
        tokenizer = AutoTokenizer.from_pretrained('ToirovSadi/t5-small')
        model.tokenizer = tokenizer
        
        return model
    
    else:
        from src.models.transformer import Transformer
        from src.models.attention import Seq2SeqAttention
        from src.models.seq2seq import Seq2Seq
        
        link = info['model_link']
        file_name = os.path.join('models',  info['file_name'])
        
        if not os.path.exists(file_name):
            gdown.download(link, file_name, quiet=False, fuzzy=True)
        
        model = torch.load(file_name, map_location='cpu')
        
        if hasattr(model, 'device'):
            model.device = 'cpu'
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'device'):
            model.encoder.device = 'cpu'
                
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'device'):
            model.decoder.device = 'cpu'

        return model

def generate(model, text, num_beams=1, model_name='t5-small'):
    if model_name == 't5-small (max_len=128)':
        # generate
        input_ids = model.tokenizer(text, return_tensors='pt').input_ids
        outputs = model.generate(input_ids, max_length=128, num_beams=num_beams)
        return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    else:
        if model_name.startswith('transformer'):
            return model.predict(text, use_beam_search=True, beam_width=num_beams)
        else: # other models don't support beam search
            return model.predict(text)

models = [
    't5-small (max_len=128)',
    'transformer1 (max_len=10)',
    'transformer2 (max_len=32)',
    'attention (max_len=10)',
    'attention2 (max_len=32)',
    'seq2seq (max_len=10)',
    'seq2seq2 (diffent decoder)',
]

# selectbox for model selection
model_name = st.sidebar.selectbox('Select model', models)
num_beams = st.sidebar.slider('num_beams', 1, 10, 3)


@st.cache_data(ttl=24*3600, max_entries=1)
def load_info():
    with open("info.json", "r") as f:
        info = json.load(f)
    return info

info = load_info()[model_name]

def get_info(model_name):
    if model_name == 't5-small (max_len=128)':
        return f"""
        #### Model Parameters:
        - max_len: {info['max_len']}
        - batch_size: {info['batch_size']}
        - num_epochs: {info['num_epochs']}
        - learning_rate: {info['learning_rate']}
        - bleu_score: {info['bleu_score']}
        
        #### Finetunned from: [t5-small]({info['source_code']})
        #### Training: [JupiterNotebook]({info['training']})
        #### Model: [GoogleDrive]({info['model_link']})
        #### Dataset: [HuggingFace]({info['dataset_link']})
        """
    else:
        return f"""
        #### Model Parameters:
        - max_len: {info['max_len']}
        - vocab_size: {info['vocab_size']}
        - batch_size: {info['batch_size']}
        - num_epochs: {info['num_epochs']}
        - num_layers: {info['num_layers']}
        - hidden_size: {info['hidden_size']}
        - dropout: {info['dropout']}
        - learning_rate: {info['learning_rate']}
        - bleu_score: {info['bleu_score']}
        
        #### Source Code: [Github]({info['source_code']})
        #### Training: [JupiterNotebook]({info['training']})
        #### Model: [GoogleDrive]({info['model_link']})
        #### Dataset: [HuggingFace]({info['dataset_link']})
        """
    

with st.sidebar.subheader('About the Model'):
    st.sidebar.write(get_info(model_name))

# text to receive text from user
text = st.text_area('Input text', 'You are welcome to try different models.')

# model will fix the text ...
model = load_model(model_name)
try:
    fixed_text = generate(model, text, num_beams=num_beams, model_name=model_name)
except Exception as e:
    st.error(e)
    fixed_text = 'Sorry, something went wrong.'

if type(fixed_text) is list:
    fixed_text = fixed_text[0]

# display the fixed text
st.text_area('fixed_text', fixed_text)