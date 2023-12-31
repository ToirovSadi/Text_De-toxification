# [Text De-toxification](https://text-de-toxification.streamlit.app)

This project helps you avoid toxic sentences. It will try to convert them to neutral. The app is deployed at Streamlit [link](https://text-de-toxification.streamlit.app) to demo.

It's more of an educational project, so you can look at implementations of models from scratch, except t5-small, which is taken from hugging-face and fine-tuned to de-toxify the text.

# How to use
## How to install
```bash
git clone https://github.com/ToirovSadi/Text_De-toxification.git

cd Text_De-toxification
pip install -r requirements.txt
```

Install all requirements for the model, to avoid any errors. It's recommented to create a new python env before installing all those packages.

## Run Streamlit Locally
```bash
streamlit run app.py
```
Then the app will open in a browser, and there you can test the app.
It will download the models that you don't have locally and store them in `models/`.

### List of available models:
- seq2seq
- seq2seq2
- attention
- attention2
- transformer
- transformer2
- t5-small
