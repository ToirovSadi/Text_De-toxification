# this is Streamlit app, for you to test the models

import streamlit as st

import torch

st.title('Text De-toxification')
st.subheader('This is a demo for text de-toxification.\nSelect a model and input a sentence to test it.')

@st.cache_data
def load_model(model_name):
    if model_name == 't5-small (max_len=128)':
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('ToirovSadi/t5-small')
        tokenizer = AutoTokenizer.from_pretrained('ToirovSadi/t5-small')
        model.tokenizer = tokenizer
        
        return model
    
    elif model_name == 'transformer1 (max_len=10)':
        from src.models.transformer import Transformer
        model = torch.load('models/transformer.01.pt', map_location='cpu')
        
        return model

def generate(model, text, num_beams=1, model_name='t5-small'):
    if model_name == 't5-small (max_len=128)':
        # generate
        return model.generate(
            input_ids=model.tokenizer.encode([text]),
            max_length=128,
            num_beams=num_beams,
            # repetition_penalty=2.5,
            # length_penalty=1.0,
            # early_stopping=True
        )
    else:
        return model.predict(text, use_beam_search=True, beam_width=num_beams)

models = [
    # 't5-small (max_len=128)',
    'transformer1 (max_len=10)',
    'transformer2 (max_len=32)',
    'attention (max_len=10)',
    'attention2 (max_len=32)',
    'seq2seq (max_len=10)',
    'seq2seq2 (diffent decoder)',
]
# selectbox for model selection
model_name = st.sidebar.selectbox('Select model', models)
num_beams = st.sidebar.slider('num_beams', 1, 10, 1)

# text to receive text from user
text = st.text_input('Input text')

# model will fix the text ...
model = load_model(model_name)
print(model.tokenizer(text))

fixed_text = generate(model, text, num_beams=num_beams, model_name=model_name)

# display the fixed text
st.write('De-toxified text:')
st.markdown(
    f'<div style="border: 2px solid #e6e6e6; border-radius: 5px; padding: 10px">{fixed_text}</div>',
    unsafe_allow_html=True,
)