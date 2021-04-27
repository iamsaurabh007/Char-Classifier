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
    
    def forward(self,anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        sim_losses = torch.relu(distance_positive - distance_negative + self.margin)
        return sim_losses.mean()

def training_step(convmodel,densemodel, batch,loss_fn,train="EMBED"):
    assert train == "EMBED", "DENSE"
    if train == "EMBED":
        anchor_images, anchor_labels,positive_images,negative_images = batch 
        embd = convmodel(anchor_images) 
        pos_embd = convmodel(positive_images) 
        neg_embd = convmodel(negative_images) 
        loss = loss_fn(embd,pos_embd,neg_embd) # Calculate loss
    if train == "DENSE":
        anchor_images,anchor_labels= batch
        embed=convmodel(anchor_images)
        out=densemodel(embed.detach())
        loss=F.cross_entropy(out,anchor_labels)
    return loss
    
def validation_step(convmodel,densemodel,batch,loss_fn):
    anchor_images, anchor_labels=batch 
    with torch.no_grad():
        embd=convmodel(anchor_images)
        out= densemodel(embd) 
        loss = loss_fn(out,anchor_labels)  # Calculate loss
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

def evaluate(convmodel,densemodel,val_loader,loss_fn):
    outputs = [validation_step(convmodel,densemodel,batch,loss_fn) for batch in val_loader]
    return validation_epoch_end(outputs)

def fit(epochs, lr, convmodel,densemodel, train_loader, val_loader,writer,opt_func):
    model_dir=join(config.MODELCHECKPOINT_PATH,"CONVPART")
    os.system('mkdir -p ' +model_dir)
    history = []
    optimizer = opt_func(convmodel.parameters(), lr, weight_decay=lr/10.0) ###CHANGES WITH TRAINING
    loss_fn=TripletLoss()     #### CHANGE UPON TRAINING PARTICULAR PART
    for epoch in tqdm(range(epochs),desc="TOTAL EPOCH"):
        # Training Phase 
        running_loss=[]
        cnt=0
        for batch in tqdm(train_loader,desc="TOTAL BATCHES"):
            optimizer.zero_grad()
            loss = training_step(convmodel,densemodel,batch,loss_fn,"EMBED")
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
        #result = evaluate(model, val_loader,loss_fn)
        #writer.add_scalar('validation loss per epoch',result['val_loss'],epoch)
        #writer.add_scalar('validation acc per epoch',result['val_acc'],epoch)
        #epoch_end(epoch, result)
        #history.append(result)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict':convmodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_mean,
                    }, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

def fit_fine(convmodel,densemodel, train_loader,optimizer):
    ls=[]
    for batch in tqdm(train_loader,desc="BATCHES FINETUNE"):
        optimizer.zero_grad()
        loss = training_step(convmodel,densemodel,batch,_,"DENSE")
        loss.backward()
        optimizer.step()
        ls.append(loss)
    return torch.stack(ls).mean()