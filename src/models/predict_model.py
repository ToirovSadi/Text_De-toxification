import gdown
import torch
import os
os.chdir("../..")
get_model_link = {
    'seq2seq': 'https://drive.google.com/file/d/1KkRqDvnjjL7mkyE4FeQkaTQjoy3NCAcP/view?usp=drive_link',
    'seq2seq2': 'https://drive.google.com/file/d/1DgBs5rPVePZxLtvekNAU40kj-3dVrJw7/view?usp=drive_link',
    'attention': 'https://drive.google.com/file/d/1WKKYRQE6xm_7FnGCSh0wyR5zkS8tyuFi/view?usp=drive_link',
    'attention2': 'https://drive.google.com/file/d/1UjnwMHGOUVTu63-RcNQnNuqqWiL2tk7a/view?usp=drive_link',
    'transformer': 'https://drive.google.com/file/d/1BRudZnO16feNYqU2K5I5w0X82Bx2qv33/view?usp=drive_link',
    'transformer2': 'https://drive.google.com/file/d/1OBMkL6tfBb6g7x-uPBu1fyeJNbRfb4na/view?usp=drive_link',
}

model_name = None
model = None

def get_model():
    global model
    if model is None:
        link = get_model_link[model_name]
        gdown.download(link, model_name + ".pt", quiet=False, fuzzy=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_name + ".pt", map_location=device)
        model.device = device

def choose_model_name():
    global model_name
    while model_name is None:
        print("Please choose one of model to make prediction")
        print("Available model names are", list(get_model.keys()))
        name = input("Model Name: ")
        if name in get_model:
            model_name = name
    
num_predictions = 10
import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str)
    args = p.parse_args()
    
    model_name = args.model_name
    if model_name is None:
        choose_model_name()
    get_model()
    
    print(f"You have {num_predictions} number of predictions per program run, you can write your sentence in console and get answer from the model")
    for _ in range(num_predictions):
        toxic_sent = input("Your sentence: ")
        try:
            
            # in later versions this `use_encoder_out` will be removed so
            # you can ignore it
            if model_name.startwith('attention'):
                print(model.predict(toxic_sent, use_encoder_out=True))
            else:
                print(model.predict(toxic_sent))
        except Exception as e:
            print("ERROR occured while trying to predict, msg:", e)
    