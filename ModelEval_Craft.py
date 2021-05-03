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

def get_ds(image, bounds):
    image= Image.open(image)
    ds=[]
    coord=[]
    labels=[]
    iouagg=[]
    #wordid=[]
    #seq=[]
    bounds=bounds['iou']
    for bound in bounds:
        if not (bound['ground'] and bound['input']):
            continue
        label=bound['text']
        iou=bound['iou']
        #wordid.append(bound['idword'])
        #seq.append(bound['sequence'])
        box = bound['input']['boundingBox']['vertices']
        x_min=min(box[0]['x'],box[1]['x'],box[2]['x'],box[3]['x'])
        x_max=max(box[0]['x'],box[1]['x'],box[2]['x'],box[3]['x'])
        y_min=min(box[0]['y'],box[1]['y'],box[2]['y'],box[3]['y'])
        y_max=max(box[0]['y'],box[1]['y'],box[2]['y'],box[3]['y'])
        
        im1 = image.crop((x_min,y_min,x_max,y_max))
        ds.append((im1,label))
        labels.append(label)
        iouagg.append(iou)
        coord.append((x_min,y_min,x_max,y_max))
    
    #image.save(str(uuid.uuid1()) + '_handwritten.png')
    return ds,coord,labels,iouagg


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
    pdf_acc=[]
    weight=[]
    mypath=join(config.pdfdata,"images")
    imgpaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    imgpaths=[str for str in imgpaths if 'average' in str]
    refinement_ratio=[0.5]
    #refinement_ratio=[0.03,0.07,0.1,0.15,0.20,0.3,0.4,0.5]
    for ref in refinement_ratio:
        coordsagg=[]
        labelsagg=[]
        pageagg=[]
        predicagg=[]
        iouagg=[]
        page_list=[]
        for imgpath in imgpaths:
            #with io.open(imgpath, 'rb') as image_file:
            #    content = image_file.read()
            jsonpath=config.pdfdata+"compare_json/"+os.path.splitext(os.path.basename(imgpath))[0]+".json"
            with open(jsonpath) as f:
                bounds = json.load(f)
            #bounds=bounds_refine(bounds,imgpath,ref)
            #print("Characters in Image=",len(bounds))
            ds,coords,labels,iou=get_ds(imgpath,bounds)
            coordsagg.extend(coords)
            labelsagg.extend(labels)
            pageagg.extend([os.path.splitext(os.path.basename(imgpath))[0]]*len(labels))
            iouagg.extend(iou)
            ds_train=DataUtils.EVALIMGDS(label_dict,ds)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            #result = ModelUtils.evaluate(model,train_gen)
            #print("Accuracy on {} page is {}".format(imgpath,result['val_acc']))
            #pdf_acc.append(len(bounds)*result['val_acc'])
            #weight.append(len(bounds))
            #os.remove(imgpath)
            #os.remove(jsonpath)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            predic=[]
            batch_accs=[]
            for batch in train_gen:
                images,labels= batch 
                with torch.no_grad(): 
                    out = model(images)
                    batch_accs.append(ModelUtils.accuracy(out,labels)) 
                    _, preds = torch.max(out, dim=1)
                predic.extend(preds.detach().cpu().numpy().tolist())
            predic=[revdict[i] for i in predic]
            predicagg.extend(predic)
            val_acc=torch.stack(batch_accs).mean()
            print("Accuracy on {} page is {}".format(imgpath,val_acc))
            pdf_acc.append(len(bounds)*val_acc)
            weight.append(len(bounds))
            page_list.append(os.path.splitext(os.path.basename(imgpath))[0])
        df_main=pd.DataFrame(list(zip(page_list,pdf_acc,weight)),columns =['Page', 'Acc','Chars'])
        df = pd.DataFrame(list(zip(coordsagg, labelsagg,predicagg,iouagg,pageagg)),\
            columns =['Coordinates', 'Actual','Predicted','IOU','Page'])
        csvpath=join(config.pdfdata,"csv/average/")
        os.system('mkdir -p ' +csvpath)
        csvpath2=join(csvpath,"MAINIncept37CRAFTFT.csv")
        csvpath=join(csvpath,"DEATAILEDIncept37CRAFTFT.csv")
        df.to_csv(csvpath,index=False)
        df_main.to_csv(csvpath2,index=False)
        print("ref={},   Accuracy Mean on this pdf is {}".format(ref,sum(pdf_acc)/sum(weight)))