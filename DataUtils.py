from PIL import Image
import numpy as np
import torchvision
import json
import torch
import utils
import random
import config

class IMGDS(torch.utils.data.Dataset):
    #Reuires a directiory with imgs and json folder in it
    def __init__(self, label_dict,root_dir,imglist,train=True):
        """
        Args:
            label_dict: mapping from labels to class
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            imglist: list of image files
        """
        self.root_dir = root_dir
        self.label_dict=label_dict
        self.images_list=imglist
        self.labels=self.loadlabel()
        self.chars_cw=[]    #classwise characters index list
        self.load_chars_list()
        self.is_train=train

    def loadimage(self,index):
        im = Image.open(self.root_dir+"/imgs/"+self.images_list[index]+".jpeg")
        if im.size[0]>im.size[1]:
            width=100
            height=int(im.size[1]*100/im.size[0])
        else:
            height=100
            width=int(im.size[0]*100/im.size[1])
        im=im.resize((width,height), Image.ANTIALIAS)
        background = Image.new('RGB', (100, 100), (255, 255, 255))
        offset = (int(round(((100 - width) / 2), 0)), int(round(((100 - height) / 2),0)))
        background.paste(im, offset)
        image=np.array(background)
        image=image/255
        image=image-1
        image=image.astype('float32')
        image=torchvision.transforms.functional.to_tensor(image)
        return image


    def loadlabel(self):
        ls=[]
        for index in range(len(self.images_list)):    
            with open(self.root_dir+"/json/"+self.images_list[index]+".json") as f:
                d= json.load(f)
                label=d['character']
                ls.append(label)
        return ls
    def load_chars_list(self):
        for i in range(len(self.label_dict)):
            p=[]
            self.chars_cw.append(p)
        for i,j in enumerate(self.labels):
            self.chars_cw[self.label_dict[j]].append(i)


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.is_train:
            anchor_image=self.loadimage(idx)
            anchor_label=self.labels[idx]
            anchor_label=self.label_dict[anchor_label]
           # positive_list = [i for i,j in enumerate(self.labels) if j==anchor_label]
            positive_item = random.choice(self.chars_cw[anchor_label])
           # negative_list = [i for i,j in enumerate(self.labels) if j!=anchor_label]
            neg_class=random.choice(range(len(self.label_dict)))
            while neg_class==anchor_label:
                neg_class=random.choice(range(len(self.label_dict)))
            negative_item = random.choice(self.chars_cw[neg_class])
            positive_image=self.loadimage(positive_item)
            negative_image=self.loadimage(negative_item)
            a=np.array(anchor_label)
            anchor_label=torch.from_numpy(a)
            return anchor_image,anchor_label,positive_image,negative_image
        else:
            anchor_image=self.loadimage(idx)
            anchor_label=self.labels[idx]
            anchor_label=self.label_dict[anchor_label]
            a=np.array(anchor_label)
            anchor_label=torch.from_numpy(a)
            return anchor_image,anchor_label

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield utils.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
class FINEIMGDS(torch.utils.data.Dataset):
    #Reuires a directiory with imgs and label in its name
    def __init__(self, label_dict,root_dir,imglist):
        """
        Args:
            label_dict: mapping from labels to class
            root_dir (string): Directory with all the images.
            imglist: list of image files
            with label class in its first two characeters of filename
        """
        self.root_dir = root_dir
        self.label_dict=label_dict
        self.images_list=imglist
        self.labels=self.loadlabel()
        
    def loadimage(self,index):
        im = Image.open(join(self.root_dir,self.images_list[index]))
        if im.size[0]>im.size[1]:
            width=100
            height=int(im.size[1]*100/im.size[0])
        else:
            height=100
            width=int(im.size[0]*100/im.size[1])
        im=im.resize((width,height), Image.ANTIALIAS)
        background = Image.new('RGB', (100, 100), (255, 255, 255))
        offset = (int(round(((100 - width) / 2), 0)), int(round(((100 - height) / 2),0)))
        background.paste(im, offset)
        image=np.array(background)
        image=image/255
        image=image-1
        image=image.astype('float32')
        image=torchvision.transforms.functional.to_tensor(image)
        return image


    def loadlabel(self):
        ls=[]
        for index in range(len(self.images_list)):    
            label=int(self.images_list[index][:2])
            ls.append(label)
        return ls


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        anchor_image=self.loadimage(idx)
        label=self.labels[idx]
        return anchor_image,label

class EVALIMGDS(torch.utils.data.Dataset):
    #Reuires a directiory with imgs and json folder in it
    def __init__(self, label_dict,ds):
        """
        Args:
        Dictionary mapping from char to class
        Ds with image crop and labels into it as tuple
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
