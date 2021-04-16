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
from tqdm import tqdm

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self,out:torch.Tensor,label:torch.Tensor, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        ce_loss=F.cross_entropy(out, label)
        sim_losses = torch.relu(distance_positive - distance_negative + self.margin)
        sim_losses=sim_losses.mean()
        losses=ce_loss+torch.mul(sim_losses,config.alpha)
        return losses

def training_step(model, batch,loss_fn):
    anchor_images, anchor_labels,positive_images,negative_images = batch 
    out,embd = model(anchor_images) 
    _,pos_embd = model(positive_images) 
    _,neg_embd = model(negative_images) 
    loss = loss_fn(out,anchor_labels,embd,pos_embd,neg_embd) # Calculate loss
    return loss
    
def validation_step(model, batch,loss_fn):
    anchor_images, anchor_labels,positive_images,negative_images = batch 
    with torch.no_grad(): 
        anchor_images, anchor_labels,positive_images,negative_images = batch 
        out,embd = model(anchor_images) 
        _,pos_embd = model(positive_images) 
        _,neg_embd = model(negative_images)  
        loss = loss_fn(out,anchor_labels,embd,pos_embd,neg_embd)  # Calculate loss
    acc = accuracy(out, anchor_labels)           # Calculate accuracy
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

def evaluate(model, val_loader,loss_fn):
    outputs = [validation_step(model,batch,loss_fn) for batch in val_loader]
    return validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader,writer,opt_func):
    model_dir=config.MODELCHECKPOINT_PATH
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=lr/10.0)
    loss_fn=TripletLoss()
    for epoch in tqdm(range(epochs),desc="TOTAL EPOCH"):
        # Training Phase 
        running_loss=[]
        cnt=0
        for batch in tqdm(train_loader,desc="TOTAL BATCHES"):
            optimizer.zero_grad()
            loss = training_step(model,batch,loss_fn)
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
        result = evaluate(model, val_loader,loss_fn)
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
