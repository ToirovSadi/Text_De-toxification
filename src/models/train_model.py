from tqdm import tqdm
import torch.nn as nn
import torch

# train the model for only one epoch
def train_epoch(model, train_dataloader, optimizer, criterion, epoch=None, clip=None, device='cpu', teacher_force=0.5):
    loop = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f'Training {epoch if epoch else ""}',
    )
    
    model.train()
    train_loss = 0
    for i, batch in loop:
        similarity, len_diff, toxic_sent, neutral_sent, toxic_val, neutral_val = batch
        toxic_sent = toxic_sent.to(device)
        neutral_sent = neutral_sent.to(device)
        
        optimizer.zero_grad()
        
        preds = model(toxic_sent, neutral_sent, teacher_force)
        # toxic_sent.shape: [num_steps, batch_size]
        # neutral_sent.shape: [num_steps, batch_size]
        # preds.shape: [num_steps, batch_size, output_dim]
        
        
        # flatten all data:  to calc the loss
        #     - neutral.shape: [num_steps * batch_size]
        #     - preds.shape: [num_steps * batch_size, output_dim]
        output_dim = preds.shape[2]
        loss = criterion(preds[1:].view(-1, output_dim), neutral_sent[1:].view(-1))
        
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()
        
        train_loss += loss.item()
        loop.set_postfix(**{"loss": train_loss / (i + 1)})
    return train_loss / len(train_dataloader)


# evaluate the model for only one epoch
def eval_epoch(model, eval_dataloader, criterion, epoch=None, device='cpu'):
    loop = tqdm(
        enumerate(eval_dataloader),
        total=len(eval_dataloader),
        desc=f'Evaluating {epoch if epoch else ""}',
    )
    
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, batch in loop:
            similarity, len_diff, toxic_sent, neutral_sent, toxic_val, neutral_val = batch
            toxic_sent = toxic_sent.to(device)
            neutral_sent = neutral_sent.to(device)

            preds = model(toxic_sent, neutral_sent, 0) # turn off the teacher force
            # toxic_sent.shape: [num_steps, batch_size]
            # neutral_sent.shape: [num_steps, batch_size]
            # preds.shape: [num_steps, batch_size, output_dim]


            # flatten all data:  to calc the loss
            #     - neutral.shape: [num_steps * batch_size]
            #     - preds.shape: [num_steps * batch_size, output_dim]
            output_dim = preds.shape[2]
            loss = criterion(preds[1:].view(-1, output_dim), neutral_sent[:1].view(-1))

            eval_loss += loss.item()
            loop.set_postfix(**{"loss": eval_loss / (i + 1)})
    return eval_loss / len(eval_dataloader)


"""
    Train the given model
    
parameters:
    * model:
        - type: str, nn.Module
    if the type is str, it's assumed that you passed path for loading the model
    and it will be loaded
    
    * loaders
        - type: list, tuple, DataLoader
    if only train dataloader is passed then no val step will be considered
    
    * optimizer
        - one of torch.optim
    if optimizer is None, then as default optimizer Adam will be taken with default parameters
    
    * criterion
        - one of the loss functions 
    if criterion is None, then as default criterion CrossEntropyLoss will be taken with default parameters
    
    * epochs
    epochs to train the model
    
    * device
    device where to train the model, if it's None, then it will get cuda if available else cpu
    
    * clip_grad
    if given then clip the grads during the training
    
    * teacher_force
    apply the teacher_force method during the training
    
    * ckpt_path
    where to save the model
    
    * best_loss
    best loss found so far
    
    * cur_epoch
    from which epoch to start
    
    * return_model
    return the trained model
"""
def train(
    model=None,
    loaders=None,
    optimizer=None,
    criterion=None,
    epochs=10,
    device=None,
    clip_grad=None,
    teacher_force=0.5,
    ckpt_path='best.pt',
    best_loss=float('inf'),
    cur_epoch=1,
    return_model=False
):
    if type(model) is str:
        # load the model
        model = None
        
    if type(loaders) not in [list, tuple]:
        loaders = [loaders]
        
    if optimizer is None:
        print("optimizer not specified using Adam with default parameters")
        optimizer = torch.optim.Adam(model.parameters())
    if criterion is None:
        print("criterion not specified using CrossEntropyLoss default parameters")
        criterion = nn.CrossEntropyLoss()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device not specified using", device)
    
    for epoch in range(cur_epoch, epochs + cur_epoch):
        train_loss = train_epoch(model, loaders[0], optimizer, criterion, epoch, clip_grad, device, teacher_force)
        if len(loaders) > 1:
            val_loss = eval_epoch(model, loaders[1], criterion, epoch, device)
        else:
            val_loss = train_loss
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
    
    if return_model:
        return best_loss, model
    return best_loss