import io
import sys 
import os
#import cv2
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import glob
from PIL import Image, ImageDraw
import config
import ModelUtils
import Resnet
import DataUtils
import InceptFC

#import pdf2image
from os import listdir
from os.path import isfile, join
import torchvision
import json
import torch
import utils
from bounds_refinement import bounds_refine
from tqdm import tqdm


def get_ds(image, bounds):
    image= Image.open(image)
    #h=img.size[1]
    #w=img.size[0]
    #w=int(2.0*w)
    #image=img.resize((w,h))
    ds=[]
    coord=[]
    labels=[]
    wordid=[]
    seq=[]
    for bound in bounds:
        label=bound['text']
        wordid.append(bound['idword'])
        seq.append(bound['sequence'])
        bound = bound['boundingBox']
        im1 = image.crop((bound["vertices"][0]['x'],
                       bound["vertices"][0]['y'],
                       bound["vertices"][2]['x'],
                       bound["vertices"][2]['y']))

        ds.append((im1,label))
        labels.append(label)
        coord.append((bound["vertices"][0]['x'],
                       bound["vertices"][0]['y'],
                       bound["vertices"][2]['x'],
                       bound["vertices"][2]['y']))
    
    #image.save(str(uuid.uuid1()) + '_handwritten.png')
    return ds


if __name__ =='__main__':
    device=config.device
    if device==None:
        device = utils.get_default_device()
    label_dict=utils.create_label_dict(config.symbols)
    revdict={}
    for i,sym in enumerate(config.symbols):
        revdict[i]=sym
    model=InceptFC.FC_Model()    
    #model=Resnet.ResNet50(3,97)
    model.to(device)
    print(config.checkpath)
    checkpoint=torch.load(config.checkpath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("MODEL LOADED")
    model.train()
    for name, child in model.named_children():
        if name in ['conv_block1',"conv_block2","conv1"]:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
        else:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=config.learning_rate, weight_decay=config.learning_rate/10)
    mypath=join(config.pdfdata,"fine_tuning_data")
    imgpaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    myvalpath="/home/ubuntu/data/ocrkdeval/good/images/"
    valid_paths = [join(myvalpath, f) for f in listdir(myvalpath) if isfile(join(myvalpath, f))]
    refinement_ratio=[0.5]
    checkpath=os.path.dirname(config.checkpath)
    checkpath=join(checkpath,"FineTune")
    os.system('mkdir -p ' +checkpath)
    p='runs/Inceptfinalrun/fine_tune/LR'+str(int(100000*l_r))+'BS'+str(batch_size)
    writer = SummaryWriter(p)

    for epoch_fine in range(config.num_epochs):
        loss_list=[]
        weight=[]
        for imgpath in tqdm(imgpaths,desc="TRAIN"):
            with io.open(imgpath, 'rb') as image_file:
                content = image_file.read()
            jsonpath=config.pdfdata+"json/"+os.path.splitext(os.path.basename(imgpath))[0]+".json"
            with open(jsonpath) as f:
                bounds = json.load(f)
            bounds=bounds_refine(bounds,imgpath,0.48)
            #print("Characters in Image=",len(bounds))
            ds=get_ds(imgpath,bounds)
            ds_train=DataUtils.IMGDS(label_dict,ds)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            result = ModelUtils.fit_fine(model,train_gen)
            loss_list.append(result.item())
            weight.append(len(bounds))

        loss_epoch=sum([weight[i]*loss_list[i] for i in range(len(weight))])/sum(weight)
        print("MEAN LOSS ON EPOCH {} is : {}".format(epoch_fine,loss_epoch))
        ## SAVE WEIGHT AFTER FINETUNE PER EPOCH
        torch.save({
                    'epoch': epoch_fine,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_epoch,
                    }, os.path.join(checkpath, 'fine-epoch-{}.pt'.format(epoch_fine)))

        ## WRITER TENSORBOARD
        writer.add_scalar('Training loss per epoch',loss_epoch,epoch_fine)
    
        ###############################################################
        ####### CHECK FOR VALIDATION+
        pdf_acc=[]
        weight=[]
        for imgpath in tqdm(valid_paths,desc="TEST"):
            with io.open(imgpath, 'rb') as image_file:
                content = image_file.read()
            jsonpath="/home/ubuntu/data/ocrkdeval/good/json/"+os.path.splitext(os.path.basename(imgpath))[0]+".json"
            with open(jsonpath) as f:
                bounds = json.load(f)
            bounds=bounds_refine(bounds,imgpath,0.48)
            #print("Characters in Image=",len(bounds))
            ds=get_ds(imgpath,bounds)
            ds_train=DataUtils.EVALIMGDS(label_dict,ds)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            result = ModelUtils.evaluate(model,train_gen)
            pdf_acc.append(len(bounds)*result['val_acc'])
            weight.append(len(bounds))
        print("EPOCHFINE={} Validation Accuracy Mean on GOOD pdf is {}".format(epoch_fine,sum(pdf_acc)/sum(weight)))
        writer.add_scalar('validation acc per epoch',sum(pdf_acc)/sum(weight),epoch_fine)
        ################################################################