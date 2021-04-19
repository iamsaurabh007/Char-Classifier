import os
import random
import config
import utils
import DataUtils
import torch
import Resnet
import InceptFC
import ModelUtils
from os import listdir
from os.path import isfile, join
from torch.utils.tensorboard import SummaryWriter

if __name__ =='__main__':
#def RUN(l_r,batch_size):
    #random.seed(10)
    dir_path=config.data_dir_path
    device=config.device
    num_epochs=config.num_epochs
    l_r=config.learning_rate
    batch_size=config.batch_size
    shuffle=config.shuffle
    num_worker=config.num_workers

    if device==None:
        device = utils.get_default_device()
    print("Device is ",device)
    label_dict=utils.create_label_dict(config.symbols)
    #imglist_train,imglist_val=utils.get_images_list(dir_path+"/imgs")
    imglist_train=utils.get_onlytrain_list(dir_path+"/imgs")
    ds_train=DataUtils.IMGDS(label_dict,dir_path,imglist_train)
    #ds_val=DataUtils.IMGDS(label_dict,dir_path,imglist_val)
    train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=batch_size,shuffle=shuffle,num_workers =num_worker,pin_memory=True)
    #valid_gen= torch.utils.data.DataLoader(ds_val,batch_size=batch_size,shuffle=shuffle,num_workers =num_worker,pin_memory=True)
    train_gen = DataUtils.DeviceDataLoader(train_gen, device)
    #valid_gen = DataUtils.DeviceDataLoader(valid_gen, device)
    #model=Resnet.ResNet50(3,config.num_classes)
    convmodel=InceptFC.Convolutional_block()
    convmodel=convmodel.to(device)
    densemodel=InceptFC.FC_block()
    densemodel=densemodel.to(device)
    p='runs/ConvPartTraining/LR'+str(int(100000*l_r))+'BS'+str(batch_size)
    writer = SummaryWriter(p)
    ModelUtils.fit(num_epochs,l_r,convmodel,densemodel,train_gen, None, opt_func=torch.optim.Adam,writer=writer)