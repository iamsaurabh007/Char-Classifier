

import torch
import torch.cuda as cuda
import torch.utils.data as data
import os
import string
from os import listdir
from os.path import isfile, join
import torchvision
from PIL import Image
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import config
import DataUtils


def training_step(model, batch):
    images, labels = batch 
    out = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss
    
def validation_step(model, batch):
    images, labels = batch 
    with torch.no_grad(): 
        out = model(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'val_loss': loss, 'val_acc': acc}
        
def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def epoch_end( epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [validation_step(model,batch) for batch in val_loader]
    return validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader,writer,opt_func):
    model_dir=config.MODELCHECKPOINT_PATH
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=lr/10.0)
    for epoch in range(epochs):
        # Training Phase 
        running_loss=[]
        cnt=0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = training_step(model,batch)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            cnt+=1
            if cnt%1000==1:
                print("MEAN LOSS TILL {} BATCH is {}".format(cnt,sum(running_loss) / len(running_loss)))
        loss_mean=sum(running_loss) / len(running_loss)
        print("Training done at epoch",epoch,"training_loss=",loss_mean)
        writer.add_scalar('training loss per epoch',loss_mean,epoch)
        # Validation phase
        result = evaluate(model, val_loader)
        writer.add_scalar('validation loss per epoch',result['val_loss'],epoch)
        writer.add_scalar('validation acc per epoch',result['val_acc'],epoch)
        epoch_end(epoch, result)
        history.append(result)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_mean,
                    }, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    return history
