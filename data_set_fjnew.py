import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random


PATH = os.path.dirname(os.path.realpath(__file__))
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


class ImageLoader(Dataset):
    def __init__(self, path, transform=None, load_type=None, choices=None):
        super(ImageLoader, self).__init__()
        self.path = path
        self.transform = transform
        self.load_type = load_type
        self.file_name_o = []
        self.file_name_f = []
        self.images_o = []
        self.images_f = []
        self.choice = choices
        if self.load_type == "train":
            dir_o = sorted(os.listdir(os.path.join(self.path, 'full_org')), key=sort_method)
            for pic in range(0, 24*50):
                temp_img = cv2.imread(os.path.join(self.path, 'full_org', dir_o[pic]))
                temp_img = cv2.resize(temp_img, (300, 150)).astype(np.float32).transpose(2, 1, 0)
                self.images_o.append(temp_img)
                self.file_name_o.append(dir_o[pic])
            dir_f = sorted(os.listdir(os.path.join(self.path, 'full_forg')), key=sort_method)
            for pic in range(0, 24 * 50):
                temp_img = cv2.imread(os.path.join(self.path, 'full_forg', dir_f[pic]))
                temp_img = cv2.resize(temp_img, (300, 150)).astype(np.float32).transpose(2, 1, 0)
                self.images_f.append(temp_img)
                self.file_name_f.append(dir_f[pic])

        elif self.load_type == "test":
            dir_o = sorted(os.listdir(os.path.join(self.path, 'full_org')), key=sort_method)
            for pic in range(50 * 24, 24 * 55):
                temp_img = cv2.imread(os.path.join(self.path, 'full_org', dir_o[pic]))
                temp_img = cv2.resize(temp_img, (300, 150)).astype(np.float32).transpose(2, 1, 0)
                self.images_o.append(temp_img)
                self.file_name_o.append(dir_o[pic])
            dir_f = sorted(os.listdir(os.path.join(self.path, 'full_forg')), key=sort_method)
            for pic in range(50 * 24, 24 * 55):
                temp_img = cv2.imread(os.path.join(self.path, 'full_forg', dir_f[pic]))
                temp_img = cv2.resize(temp_img, (300, 150)).astype(np.float32).transpose(2, 1, 0)
                self.images_f.append(temp_img)
                self.file_name_f.append(dir_f[pic])

    def __getitem__(self, idx):
        a = self.choice[idx]
        if self.choice[idx][2] == 1:
            ref = self.images_o[self.choice[idx][0] + int(idx/552) * 24]
            ref = torch.from_numpy(ref)
            ref_path = self.file_name_o[self.choice[idx][0] + int(idx/552) * 24]
            test = self.images_o[self.choice[idx][1] + int(idx/552) * 24]
            test = torch.from_numpy(test)
            test_path = self.file_name_o[self.choice[idx][1] + int(idx / 552) * 24]
            ground_truth = 1
        else:
            ref = self.images_o[self.choice[idx][0] + int(idx/552) * 24]
            ref = torch.from_numpy(ref)
            ref_path = self.file_name_o[self.choice[idx][0] + int(idx / 552) * 24]
            test = self.images_f[self.choice[idx][1] + int(idx/552) * 24]
            test = torch.from_numpy(test)
            test_path = self.file_name_f[self.choice[idx][1] + int(idx / 552) * 24]
            ground_truth = 0
        # print("ref_path",ref_path)
        # print('test_path',test_path)
        return {'ref_img': ref.float().to(device), 'test_img': test.float().to(device), 'ground_truth': torch.tensor(ground_truth).float().to(device), 'ref_path': ref_path, 'test_path': test_path}

    def __len__(self):
        return len(self.choice)


if __name__ == '__main__':

    train_pair = []
    test_pair = []

    for i in range(0, 50):
        org_org = [(j, k, 1) for j in range(0, 24) for k in range(j+1, 24)]
        train_pair.extend(org_org)
        org_forg = [(j, k, 0) for j in range(0, 24) for k in range(0, 24)]
        o_f = random.sample(org_forg, k=276)
        train_pair.extend(o_f)
    for i in range(50, 55):
        org_org = [(j, k, 1) for j in range(0, 24) for k in range(j+1, 24)]
        test_pair.extend(org_org)
        org_forg = [(j, k, 0) for j in range(0, 24) for k in range(0, 24)]
        o_f = random.sample(org_forg, k=276)
        test_pair.extend(o_f)

    trainData = ImageLoader(os.path.join(PATH, 'dataset_cedar'), load_type="train", transform=None, choices=train_pair)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=8, shuffle=True, num_workers=0)

    testData = ImageLoader(os.path.join(PATH, 'dataset_cedar'), load_type="test", transform=None, choices=test_pair)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=8, shuffle=False, num_workers=0)

    for i, data in enumerate(trainLoader, 0):
        images = data['ref_img']
        images1 = data['test_img']
        g_truth = data['ground_truth']
        images_path = data['ref_path']
        images1_path = data['test_path']
        a = 0
    print("end")
