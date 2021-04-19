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
    return ds,coord,labels,wordid,seq


class IMGDS(torch.utils.data.Dataset):
    #Reuires a directiory with imgs and json folder in it
    def __init__(self, label_dict,ds):
        """
        Args:
            label_dict: mapping from labels to class
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            imglist: list of image files
        """
        #self.root_dir = root_dir
        self.label_dict=label_dict
        self.ds=ds

    def loadimage(self,index):
        im = self.ds[index][0]
        if im.size[0]>im.size[1]:
            width=100
            height=int(im.size[1]*100/im.size[0])
        else:
            height=100
            width=int(im.size[0]*100/im.size[1])
        try:
            im=im.resize((width,height), Image.ANTIALIAS)
        except:
            print(im.size[0],im.size[1])
        background = Image.new('RGB', (100, 100), (255, 255, 255))
        offset = (int(round(((100 - width) / 2), 0)), int(round(((100 - height) / 2),0)))
        background.paste(im, offset)
        image=np.array(background)
        image=image/255
        image=image-1
        image=image.astype('float32')
        image=torchvision.transforms.functional.to_tensor(image)
        return image


    def loadlabel(self,index):
        label=self.ds[index][1]
        try:
            p=self.label_dict[label]
        except:
            p=96
        a=np.array(p)
        a=torch.from_numpy(a)
        return a
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image=self.loadimage(idx)
        label=self.loadlabel(idx)
        return image,label

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
    refinement_ratio=[0.5]
    #refinement_ratio=[0.03,0.07,0.1,0.15,0.20,0.3,0.4,0.5]
    for ref in range(1):
        coordsagg=[]
        labelsagg=[]
        pageagg=[]
        predicagg=[]
        wordidagg=[]
        sequenceagg=[]
        page_list=[]
        for imgpath in imgpaths:
            with io.open(imgpath, 'rb') as image_file:
                content = image_file.read()
            jsonpath=config.pdfdata+"json/"+os.path.splitext(os.path.basename(imgpath))[0]+".json"
            with open(jsonpath) as f:
                bounds = json.load(f)
            bounds=bounds_refine(bounds,imgpath,0.48)
            #print("Characters in Image=",len(bounds))
            ds,coords,labels,wordid,seq=get_ds(imgpath,bounds)
            coordsagg.extend(coords)
            labelsagg.extend(labels)
            pageagg.extend([os.path.splitext(os.path.basename(imgpath))[0]]*len(labels))
            wordidagg.extend(wordid)
            sequenceagg.extend(seq)
            ds_train=IMGDS(label_dict,ds)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
           # result = ModelUtils.evaluate(model,train_gen)
            #os.remove(imgpath)
            #os.remove(jsonpath)
            train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=64,shuffle=False,num_workers =6,pin_memory=True)
            train_gen =DataUtils.DeviceDataLoader(train_gen, device)
            predic=[]
            batch_accs=[]
            for batch in train_gen:
                images,labels= batch 
                with torch.no_grad(): 
                    out,_ = model(images)
                    batch_accs.append(ModelUtils.accuracy(out,labels))
                    val_acc=torch.stack(batch_accs).mean() 
                    _, preds = torch.max(out, dim=1)
                predic.extend(preds.detach().cpu().numpy().tolist())
            print("Accuracy on {} page is {}".format(imgpath,val_acc))
            pdf_acc.append(len(bounds)*val_acc)
            weight.append(len(bounds))
            predic=[revdict[i] for i in predic]
            predicagg.extend(predic)
            page_list.append(os.path.splitext(os.path.basename(imgpath))[0])
        df_main=pd.DataFrame(list(zip(page_list,pdf_acc,weight)),columns =['Page', 'Acc','Chars'])
        df = pd.DataFrame(list(zip(coordsagg, labelsagg,predicagg,wordidagg,sequenceagg,pageagg)),\
           columns =['Coordinates', 'Actual','Predicted','Word','Sequence','Page'])
        csvpath=join(config.pdfdata,"csv/")
        os.system('mkdir -p ' +csvpath)
        csvpath2=join(csvpath,"MAINIncept161RefineBinaryTripletkdeval.csv")
        csvpath=join(csvpath,"DEATAILEDIncept161RefineBinaryTripletkdeval.csv")
        df.to_csv(csvpath,index=False)
        df_main.to_csv(csvpath2,index=False)
        print("ref={},   Accuracy Mean on this pdf is {}".format(ref,sum(pdf_acc)/sum(weight)))
