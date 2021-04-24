from PIL import Image
import numpy as np
import torchvision
import json
import torch
import utils
from os.path import join


class IMGDS(torch.utils.data.Dataset):
    #Reuires a directiory with imgs and json folder in it
    def __init__(self, label_dict,root_dir,imglist):
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


    def loadlabel(self,index):
        with open(self.root_dir+"/json/"+self.images_list[index]+".json") as f:
            d= json.load(f)
            label=d['character']
            a=np.array(self.label_dict[label])
            a=torch.from_numpy(a)
            return a
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image=self.loadimage(idx)
        label=self.loadlabel(idx)
        return image,label


class FINEIMGDS(torch.utils.data.Dataset):
    #Reuires a list of files of images with labels in name
    def __init__(self, label_dict,root_dir,imglist):
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


    def loadlabel(self,index):
        label=int(self.images_list[index][:2])
        a=np.array(label)
        a=torch.from_numpy(a)
        return a
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image=self.loadimage(idx)
        label=self.loadlabel(idx)
        return image,label


class EVALIMGDS(torch.utils.data.Dataset):
    #Reuires a  list with PIL image and respective labels, best for page inference
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
        try:
            if im.size[0]>im.size[1]:
                width=100
                height=int(im.size[1]*100/im.size[0])
            else:
                height=100
                width=int(im.size[0]*100/im.size[1])
            im=im.resize((width,height), Image.ANTIALIAS)
        except:
            print(index,im.size[0],im.size[1])
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
