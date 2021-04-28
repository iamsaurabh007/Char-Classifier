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
import csv

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_label_dict(symbols):
    #symbols=list(string.printable[:94])
    #ymbols.append(u"\u00A9")
    #symbols.append(u"\u2122")
    #symbols.append(" ")
    label={}
    for i,sym in enumerate(symbols):
        label[sym]=i
    print("Dictionary Created with {} symbols".format(len(symbols)))
    return label

def get_images_list(mypath,number=None):
    #Currently jpeg implementation only
    onlyfiles = [f[:-5] for f in listdir(mypath) if isfile(join(mypath, f))]
    random.shuffle(onlyfiles)
    if number:
        onlyfiles=onlyfiles[:number]   
    print("Images Available Train={}, Valid={} ".format(int(0.95*len(onlyfiles)),int(0.05*len(onlyfiles)))) 
    return onlyfiles[:int(0.95*len(onlyfiles))],onlyfiles[int(0.95*len(onlyfiles)):]
def get_onlytrain_list(mypath,number=None):
    onlyfiles = [f[:-5] for f in listdir(mypath) if isfile(join(mypath, f))]
    random.shuffle(onlyfiles)
    if number:
        onlyfiles=onlyfiles[:number]   
    print("Images Available Train={}".format(len(onlyfiles)))
    return onlyfiles

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def csv_to_ls(path):
    ls=[]
    with open(path,newline='') as f:
        reader = csv.reader(f)
        ls=[row[0] for row in reader]
    return ls


def get_ds_vision(image,bounds):
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

    return ds

def get_ds_crafts(image, bounds):
    image= Image.open(image)
    ds=[]
    bounds=bounds['iou']
    for bound in bounds:
        if not (bound['ground'] and bound['input']):
            continue
        label=bound['text']
        box = bound['input']['boundingBox']['vertices']
        x_min=min(box[0]['x'],box[1]['x'],box[2]['x'],box[3]['x'])
        x_max=max(box[0]['x'],box[1]['x'],box[2]['x'],box[3]['x'])
        y_min=min(box[0]['y'],box[1]['y'],box[2]['y'],box[3]['y'])
        y_max=max(box[0]['y'],box[1]['y'],box[2]['y'],box[3]['y'])
        
        im1 = image.crop((x_min,y_min,x_max,y_max))
        ds.append((im1,label))
        
    return ds