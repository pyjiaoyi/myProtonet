import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms

import PIL.Image as Image
import numpy as np


def transform_data(ds):
    ds=np.array(ds,np.float(32),copy=False)
    ds=torch.from_numpy(ds).reshape(ds.size[0],ds.size[1],ds.size[4],ds.size[2],-1)

    return ds


def extract_data(split,data_dir):
    san_split=["train","trainval","test"]
    if split not in san_split:
        raise ValueError("Invalid data split! Please input 'train' or 'trainval'")
    
    mini_imagenet={"train":None,"val":None,"test":None}
    data_path=[]
    if split=="test":
        data_path.append(os.path.join(data_dir,"{:s}.csv".format(split)))
    

    else:
        data_path.append(os.path.join(data_dir,"train.csv"))
        data_path.append(os.path.join(data_dir,"val.csv"))

        total_dataset=[]
        for dp in data_path:
            with open(dp,'r') as f:
                total_dataset.append(list(f)[1:])
        
        mini_imagenet["train"]={}

        if split=="train":
            mini_imagenet["val"]={}

            for data_label in total_dataset[0]:
                data,label=data_label.split(",")
                if label not in mini_imagenet["train"].keys():
                    mini_imagenet["train"][label]=[]
                mini_imagenet["train"][label].append(data_dir+"images"+data)

            for data_label in total_dataset[1]:
                data,label=data_label.split(",")
                if label not in mini_imagenet["val"].keys():
                    mini_imagenet["val"][label]=[]
                mini_imagenet["val"][label].append(data_dir+"images"+data)

        else:
            for i in total_dataset:
                for data_label in i:
                    data,label=data_lable.split(",")
                    if label not in mini_imagenet["train"].keys():
                        mini_imagenet["train"][label]=[]
                    mini_imagenet["train"][label].append(data_dir+"images"+data)



    return mini_imagenet


class miniImagenet(Dataset):
    def __init__(self,path_dict,transform,res_h,res_w,n_shot):
        self.path_dict=path_dict
        self.transform=transform
        self.data=[]
        self.n_shot=n_shot
        for i in path_dict.key():
            dt=[]
            for p in path_dict[i]:
                img=Image.open(p)
                img.resize((res_w,res_h))
                dt.append(img)
            self.data.append(dt)
        self.data=transform(self.data)



    def __len__(self):
        return self.path_dict.shape[0]

    def __getitem__(self,key):
        dl=self.data[key].shape[0]
        return self.data[key][torch.randperm(dl)[:self.n_shot]]


class few_shot_sampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes=n_classes
        self.n_way=n_way
        self.n_episodes=n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        class_label=torch.randperm(self.n_classes)[0:self.n_way]
        return class_label
