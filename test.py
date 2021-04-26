
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
        if xmax-xmin==0 or ymax-ymin==0:
            continue
        im1 = image.crop((xmin,ymin,xmax,ymax))
        ds.append((im1,label))
        
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
    for model_wt in config.testweights:  
        model=InceptFC.FC_Model()    
        #model=Resnet.ResNet50(3,97)
        model.to(device)
        checkpath=join(config.weightfilepath,model_wt)
        print(checkpath)
        checkpoint=torch.load(checkpath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("MODEL LOADED")
        model.train()
        for myvalpath in config.testfiles:
            print("Model Weight is ",model_wt)
            print("Test file is ",myvalpath)
            valpath=join(myvalpath,"images")
            valid_paths = [join(valpath, f) for f in listdir(valpath) if isfile(join(valpath, f))]
            pdf_acc=[]
            weight=[]
            for imgpath in valid_paths:
                with io.open(imgpath, 'rb') as image_file:
                    content = image_file.read()
                jsonpath=join(myvalpath,"json")+os.path.splitext(os.path.basename(imgpath))[0]+".json"
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
            print("Test Accuracy Mean on this pdf is {}".format(sum(pdf_acc)/sum(weight)))
            ################################################################

