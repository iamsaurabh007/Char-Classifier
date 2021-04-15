from PIL import Image
import numpy as np
import torchvision
import json
import torch
import utils


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

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if is_train:
            anchor_image=self.loadimage(idx)
            anchor_label=self.labels[idx]
            positive_list = [i for i,j in enumerate(self.labels) if j==anchor_label]
            positive_item = random.choice(positive_list)
            neagtive_list = [i for i,j in enumerate(self.labels) if j!=anchor_label]
            negative_item = random.choice(negative_list)
            positive_image=self.loadimage(positive_item)
            negative_image=self.loadimage(negative_item)
            a=np.array(self.label_dict[anchor_label])
            anchor_label=torch.from_numpy(a)
            return anchor_image,anchor_label,positive_image,negative_image
        else:
            anchor_image=self.loadimage(idx)
            return anchor_image

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
