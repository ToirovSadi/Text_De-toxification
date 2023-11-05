# Text De-toxification
## Name Surname: Sadi Toirov, group B21-AI-01

This project helps you to avoid toxic sentences. It will try to convert them to neutral.

# How to use
## How to install
```bash
git clone https://github.com/ToirovSadi/Text_De-toxification.git

cd Text_De-toxification
pip install -r requirements.txt
```

Install all requirements for the model, to avoid any errors. It's recommented to create a new python env before installing all those packages.


## How to predict
```bash
python src/models/predict_model.py --model_name='transformer'
```
or 
```bash
python src/models/predict_model.py
```
and then through console type the name of the model

### List of available models:
- seq2seq
- seq2seq2
- attention
- attention2
- transformer
- transformer2
