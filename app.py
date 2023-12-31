# this is Streamlit app, for you to test the models

import streamlit as st
import torch
import gdown
import json
import os

st.title('Text De-toxification')
st.subheader('This is a demo for text de-toxification.\nSelect a model and input a sentence to test it.')

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
            gdown.download(link, file_name, quiet=False, fuse=True)
        
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

with open("info.json", "r") as f:
    info = json.load(f)[model_name]

def get_info(model_name, p):
    if model_name == 't5-small (max_len=128)':
        return f"""
        #### Model Parameters:
        - max_len: {p['max_len']}
        - batch_size: {p['batch_size']}
        - num_epochs: {p['num_epochs']}
        - learning_rate: {p['learning_rate']}
        - bleu_score: {p['bleu_score']}
        
        #### Finetunned from: [t5-small]({p['source_code']})
        #### Training: [JupiterNotebook]({p['training']})
        #### Model: [GoogleDrive]({p['model_link']})
        #### Dataset: [HuggingFace]({p['dataset_link']})
        """
    else:
        return f"""
        #### Model Parameters:
        - max_len: {p['max_len']}
        - vocab_size: {p['vocab_size']}
        - batch_size: {p['batch_size']}
        - num_epochs: {p['num_epochs']}
        - num_layers: {p['num_layers']}
        - hidden_size: {p['hidden_size']}
        - dropout: {p['dropout']}
        - learning_rate: {p['learning_rate']}
        - bleu_score: {p['bleu_score']}
        
        #### Source Code: [Github]({p['source_code']})
        #### Training: [JupiterNotebook]({p['training']})
        #### Model: [GoogleDrive]({p['model_link']})
        #### Dataset: [HuggingFace]({p['dataset_link']})
        """
    

with st.sidebar.subheader('About the Model'):
    st.sidebar.write(get_info(model_name, info))

# text to receive text from user
text = st.text_area('Input text', 'You are welcome to try different models.')

# model will fix the text ...
model = load_model(model_name)
fixed_text = generate(model, text, num_beams=num_beams, model_name=model_name)

if type(fixed_text) is list:
    fixed_text = fixed_text[0]

# display the fixed text
st.text_area('fixed_text', fixed_text)