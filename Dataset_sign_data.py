import torch
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader,Dataset
from glob import glob
import random

class WritingDataset(Dataset):
    def __init__(self, dataset_folder, img_size, mode='train'):
        self.mode = mode
        self.img_size = img_size
        if self.mode =='train':
            self.idx = range(1,70)
        elif self.mode =='test':
            self.idx = range(49,70)
        else:
            print("Invalid mode", mode)


        self.data = {}
        for i in self.idx:
            self.data[i] = {}
            self.data[i]['real'] = sorted(glob('{}/train/{:0>3d}/*'.format(dataset_folder, i)))
            self.data[i]['forg'] = sorted(glob('{}/train/{:0>3d}_forg/*'.format(dataset_folder, i)))


    def __len__(self):
        return len(self.data.keys())
    
    def __getitem__(self, index):
        group = self.data[index+1]
        real, forg = group['real'], group['forg']

        label = np.random.randint(0,2)
        if label == 0:
            real_sample, forg_sample = random.sample(real,1), random.sample(forg,1)
            print(real_sample)
            real_sample, forg_sample =cv2.imread(real_sample[0]),cv2.imread(forg_sample[0])
            real_sample=cv2.resize(real_sample,self.img_size).swapaxes(0,2)
            real_sample=torch.tensor(real_sample).float()
            forg_sample=cv2.resize(forg_sample,self.img_size).swapaxes(0,2)
            forg_sample = torch.tensor(forg_sample).float()

            return [real_sample, forg_sample, label]
        else:
            real_sample, real_sample2 = random.sample(real,1), random.sample(real,1)
            print(real_sample)
            real_sample, real_sample2 =cv2.imread(real_sample[0]),cv2.imread(real_sample2[0])
            real_sample=cv2.resize(real_sample,self.img_size).swapaxes(0,2)
            real_sample=torch.tensor(real_sample).float()
            real_sample2=cv2.resize(real_sample2,self.img_size).swapaxes(0,2)
            real_sample2 = torch.tensor(real_sample2).float()
            return [real_sample, real_sample2, label]





        