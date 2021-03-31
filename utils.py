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