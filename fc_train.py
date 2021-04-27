import io
import json
import os
import random
import sys
#import pdf2image
from os import listdir
from os.path import isfile, join

#import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
#import matplotlib.pyplot as plt
#import glob
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import DataUtils
import InceptFC
import ModelUtils
import Resnet
import utils
from bounds_refinement import bounds_refine


def get_ds(image,bounds):
    image= Image.open(image)
    #h=img.size[1]
    #w=img.size[0]
    #w=int(2.0*w)
    #image=img.resize((w,h))
    ds=[]
 
    for bound in bounds:
        label=bound['text']
        bound = bound['boundingBox']
        xmin=min(bound["vertices"][0]['x'],bound["vertices"][1]['x'],bound["vertices"][2]['x'],bound["vertices"][3]['x'])
        xmax=max(bound["vertices"][0]['x'],bound["vertices"][1]['x'],bound["vertices"][2]['x'],bound["vertices"][3]['x'])
        ymin=min(bound["vertices"][0]['y'],bound["vertices"][1]['y'],bound["vertices"][2]['y'],bound["vertices"][3]['y'])
        ymax=max(bound["vertices"][0]['y'],bound["vertices"][1]['y'],bound["vertices"][2]['y'],bound["vertices"][3]['y'])
        if xmax-xmin<=1 or ymax-ymin<=1:
            continue
        im1 = image.crop((xmin,ymin,xmax,ymax))
        ds.append((im1,label))
        
    #image.save(str(uuid.uuid1()) + '_handwritten.png')
    return ds

if __name__ =='__main__':
    batchsize=config.batch_size
    lr=config.learning_rate
    num_epochs=config.num_epochs
    device=config.device
    if device==None:
        device = utils.get_default_device()
    label_dict=utils.create_label_dict(config.symbols)
    revdict={}
    for i,sym in enumerate(config.symbols):
        revdict[i]=sym
    convmodel=InceptFC.Convolutional_block()
    densemodel=InceptFC.FC_block()    
    #model=Resnet.ResNet50(3,97)
    convmodel.to(device)
    densemodel.to(device)
    print(config.checkpath)
    checkpoint=torch.load(config.checkpath, map_location=device)
    convmodel.load_state_dict(checkpoint['model_state_dict'])
    print("MODEL LOADED")
    convmodel.train()

    for name, child in convmodel.named_children():
        print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, densemodel.parameters()),lr=lr, weight_decay=lr/10.)
    finepath=config.data_dir_path
    myvalpath="/home/ubuntu/data/ocr/kdeval/good/images/"
    valid_paths = [join(myvalpath, f) for f in listdir(myvalpath) if isfile(join(myvalpath, f))]
    refinement_ratio=[0.5]
    checkpath=os.path.dirname(config.MODELCHECKPOINT_PATH)
    checkpath=join(checkpath,"FC_PART")
    os.system('mkdir -p ' +checkpath)
    p='runs/FC_PART_TRAINING/LR'+str(int(1000000*lr))+'BS'+str(batchsize)
    writer = SummaryWriter(p)
    fineds=[f for f in listdir(finepath) if isfile(join(finepath, f))]
    for epoch_fine in range(num_epochs):
        random.shuffle(fineds)
        ds_train=DataUtils.FINEIMGDS(label_dict,finepath,fineds)
        train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=batchsize,shuffle=True,num_workers =6,pin_memory=True)
        train_gen =DataUtils.DeviceDataLoader(train_gen, device)
        result = ModelUtils.fit_fine(convmodel,densemodel,train_gen,optimizer)
        loss_epoch=result.item()
        print("MEAN LOSS ON EPOCH {} is : {}".format(epoch_fine,loss_epoch))
        ## SAVE WEIGHT AFTER FINETUNE PER EPOCH

        torch.save({
                    'epoch': epoch_fine,
                    'model_state_dict': densemodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_epoch,
                    }, os.path.join(checkpath, 'fc-epoch-{}.pt'.format(epoch_fine)))
    
        ## WRITER TENSORBOARD
        writer.add_scalar('Training loss per epoch',loss_epoch,epoch_fine)
        ###############################################################
        ####### CHECK FOR VALIDATION+
        pdf_acc=[]
        weight=[]
        for imgpath in tqdm(valid_paths,desc="TEST"):
            with io.open(imgpath, 'rb') as image_file:
                content = image_file.read()
            jsonpath="/home/ubuntu/data/ocr/kdeval/good/json/"+os.path.splitext(os.path.basename(imgpath))[0]+".json"
            with open(jsonpath) as f:
                bounds = json.load(f)
            bounds=bounds_refine(bounds,imgpath,0.48)
            #print("Characters in Image=",len(bounds))
            ds=get_ds(imgpath,bounds)
            ds_train=DataUtils.EVALIMGDS(label_dict,ds)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            batch_accs=[]
            for batch in train_gen:
                images,labels= batch 
                with torch.no_grad(): 
                    embed=convmodel(images)
                    out=densemodel(embed.detach())
                    batch_accs.append(ModelUtils.accuracy(out,labels))
            val_acc=torch.stack(batch_accs).mean().item() 
            pdf_acc.append(len(bounds)*val_acc)
            weight.append(len(bounds))
        print("EPOCHFINE={} Validation Accuracy Mean on GOOD pdf is {}".format(epoch_fine,sum(pdf_acc)/sum(weight)))
        writer.add_scalar('validation acc per epoch',sum(pdf_acc)/sum(weight),epoch_fine)
        ################################################################
