from torch.utils.data import DataLoader,Dataset
import cv2
import numpy as np
import torch
import os
import glob
import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_method(elem):
    bound = [0, 0, 0]
    for c in range(0, len(elem)):
        if elem[c] == '_':
            bound[0] = c + 1
            break
    for c in range(bound[0], len(elem)):
        if elem[c] == '_':
            bound[1] = c
            break
    for c in range(bound[1]+1, len(elem)):
        if elem[c] == '.':
            bound[2] = c
            break
    a = int(elem[bound[0]: bound[1]])
    b = int(elem[(bound[1]+1): bound[2]])
    return a*24+b

class dataset(Dataset):
    def __init__(self,path = "DATA",data_shape=None,mode = "train"):
        forgs =  sorted(os.listdir(os.path.join(path,"signatures","full_forg")), key=sort_method)
        orgs =  sorted(os.listdir(os.path.join(path,"signatures","full_org")), key=sort_method)
        forg_images,org_images = [],[]
        forg_paths,org_paths = [],[]
        for i in range(24 * 55):
            image_f = cv2.imread(os.path.join(os.path.join(path,"signatures","full_forg"),forgs[i]))
            image_f = cv2.resize(image_f, data_shape).astype(np.float32).transpose(2,1,0)
            forg_images.append(image_f)
            forg_paths.append(os.path.join(os.path.join(path,"signatures","full_forg"),forgs[i]))
            image_t = cv2.imread(os.path.join(os.path.join(path,"signatures","full_org"),orgs[i]))
            image_t = cv2.resize(image_t, data_shape).astype(np.float32).transpose(2, 1, 0)
            org_images.append(image_t)
            org_paths.append(os.path.join(os.path.join(path,"signatures","full_org"),orgs[i]))
        self.Combs = []

        if mode == "train":
            for i in range(50):  # for each one in 55 kinds of names
                counter = 0
                temp_list = []
                for m in range(24):
                    for n in range(m,24):
                        counter+=1
                        self.Combs.append((org_images[i * 24 + m], org_images[i * 24 + n], 1,org_paths[i * 24 + m],org_paths[i * 24 + n]))

                
                for j in range(24):
                    for k in range(24):
                        temp_list.append((org_images[i * 24 + j], forg_images[i * 24 + k], 0,org_paths[i * 24 + j],forg_paths[i * 24 + k]))
                
                random.shuffle(temp_list)
                temp_list = temp_list[:counter]
                for temp_item in temp_list:
                    self.Combs.append(temp_item)

        else:
            for i in range(50,55):  # for each one in 55 kinds of names
                counter = 0
                temp_list = []
                for m in range(24):
                    for n in range(m,24):
                        counter+=1
                        self.Combs.append((org_images[i * 24 + m], org_images[i * 24 + n], 1,org_paths[i * 24 + m],org_paths[i * 24 + n]))

                for j in range(24):
                    for k in range(24):
                        temp_list.append((org_images[i * 24 + j], forg_images[i * 24 + k], 0,org_paths[i * 24 + j],forg_paths[i * 24 + k]))
                
                random.shuffle(temp_list)
                temp_list = temp_list[:counter]
                for temp_item in temp_list:
                    self.Combs.append(temp_item)






    def __getitem__(self, index):
        img1,img2,gt,imgpath1,imgpath2 = self.Combs[index]
        data = {
            "img1":torch.from_numpy(img1).float().to(device),
            "img2":torch.from_numpy(img2).float().to(device),
            "gt": torch.tensor([gt]).float().to(device)
        }
        # print("imgpath1: ",imgpath1)
        # print("imgpath2: ",imgpath2)
        # print("gt",gt)
        return data

    def __len__(self):
        return len(self.Combs)



