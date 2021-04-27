
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
        for category in config.testfiles:   #########
            print("Model Weight is ",model_wt)
            print("Test file is ",category)
            myvalpath=config.testpath
            valpath=join(myvalpath,"images/")
            valid_paths = [join(valpath, f) for f in listdir(valpath) if isfile(join(valpath, f))]
            valid_paths=[str for str in valid_paths if category in str]   #####
            pdf_acc=[]
            weight=[]
            for imgpath in valid_paths:
                with io.open(imgpath, 'rb') as image_file:
                    content = image_file.read()
                jsonpath=join(myvalpath,"compare_json/")+os.path.splitext(os.path.basename(imgpath))[0]+".json"
                with open(jsonpath) as f:
                    bounds = json.load(f)
                bounds=bounds_refine(bounds,imgpath,0.48)
                #print("Characters in Image=",len(bounds))
                ds=utils.get_ds_crafts(imgpath,bounds)
                ds_train=DataUtils.EVALIMGDS(label_dict,ds)
                train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
                train_gen =DataUtils.DeviceDataLoader(train_gen, device)
                result = ModelUtils.evaluate(model,train_gen)
                pdf_acc.append(len(bounds)*result['val_acc'])
                weight.append(len(bounds))
            print("Test Accuracy Mean on this pdf is {}".format(sum(pdf_acc)/sum(weight)))
            print("/n")
            ################################################################

