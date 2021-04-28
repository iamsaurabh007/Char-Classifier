"""
On list of model weights,
On list of files folder,
"""
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
from torch.utils.tensorboard import SummaryWriter
import random



if __name__ =='__main__':
    device=config.device
    if device==None:
        device = utils.get_default_device()
    label_dict=utils.create_label_dict(config.symbols)
    revdict={}
    for i,sym in enumerate(config.symbols):
        revdict[i]=sym

    convmodel=InceptFC.Convolutional_block()
    convmodel.to(device)
    checkpoint=torch.load(config.conv_checkpath, map_location=device)
    convmodel.load_state_dict(checkpoint['model_state_dict'])
    convmodel.train()
    for model_wt in config.testweights:  
        densemodel=InceptFC.FC_block()    
        #model=Resnet.ResNet50(3,97)
        densemodel.to(device)
        checkpath=join(config.weightfilepath,model_wt)
        print(checkpath)
        checkpoint=torch.load(checkpath, map_location=device)
        densemodel.load_state_dict(checkpoint['model_state_dict'])
        print("MODEL LOADED")
        densemodel.train()
        for myvalpath in config.testfiles:      ####
            print("Model Weight is ",model_wt)
            print("Test file is ",myvalpath)  ##
            #myvalpath=config.testpath   ###
            valpath=join(myvalpath,"images/")
            valid_paths = [join(valpath, f) for f in listdir(valpath) if isfile(join(valpath, f))]
            #valid_paths=[str for str in valid_paths if category in str]   #####
            pdf_acc=[]
            weight=[]
            for imgpath in valid_paths:
                jsonpath=join(myvalpath,"json/")+os.path.splitext(os.path.basename(imgpath))[0]+".json"  ###
                with open(jsonpath) as f:
                    bounds = json.load(f)
                bounds=bounds_refine(bounds,imgpath,0.48)
                #print("Characters in Image=",len(bounds))
                ds=utils.get_ds_vision(imgpath,bounds)
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
            print("Test Accuracy Mean on this pdf is {}".format(sum(pdf_acc)/sum(weight)))
            print("/n")
            ################################################################
