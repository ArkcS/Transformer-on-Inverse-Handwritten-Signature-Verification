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
        self.pairs = []
        if self.mode == 'train':
            self.idx = range(1, 60)
            for i in self.idx:
                choices = []
                img_path = sorted(glob('{}/train/{:0>3d}/*'.format(dataset_folder, i)))
                img_path_f = sorted(glob('{}/train/{:0>3d}_forg/*'.format(dataset_folder, i)))

                org_org = [(j, k, 1) for j in range(0, len(img_path)) for k in range(j + 1, len(img_path))]
                org_forg = [(j, k, 0) for j in range(0, len(img_path)) for k in range(0, len(img_path_f))]

                if len(org_org) <= len(org_forg):
                    choices.extend(org_org)
                    o_f = random.sample(org_forg, k=len(org_org))
                    choices.extend(o_f)
                else:
                    o_o = random.sample(org_org, k=len(org_forg))
                    choices.extend(o_o)
                    choices.extend(org_forg)

                for each in choices:
                    if each[2] == 1:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 1))
                    else:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path_f[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 0))

        elif self.mode == 'test':
            self.idx = range(60, 70)
            for i in self.idx:
                choices = []
                img_path = sorted(glob('{}/test/{:0>3d}/*'.format(dataset_folder, i)))
                img_path_f = sorted(glob('{}/test/{:0>3d}_forg/*'.format(dataset_folder, i)))

                org_org = [(j, k, 1) for j in range(0, len(img_path)) for k in range(j + 1, len(img_path))]
                org_forg = [(j, k, 0) for j in range(0, len(img_path)) for k in range(0, len(img_path_f))]

                if len(org_org) <= len(org_forg):
                    choices.extend(org_org)
                    o_f = random.sample(org_forg, k=len(org_org))
                    choices.extend(o_f)
                else:
                    o_o = random.sample(org_org, k=len(org_forg))
                    choices.extend(o_o)
                    choices.extend(org_forg)

                for each in choices:
                    if each[2] == 1:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 1))
                    else:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path_f[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 0))
        else:
            print("Invalid mode", mode)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        ref = self.pairs[index][0]
        ref = torch.from_numpy(ref)
        test = self.pairs[index][1]
        test = torch.from_numpy(test)
        ground_truth = torch.FloatTensor([self.pairs[index][2]])
        return {'ref_img': ref, 'test_img': test, 'ground_truth': ground_truth}


if __name__ == '__main__':
    train_data = WritingDataset(dataset_folder='sign_data', img_size=(300, 150), mode='train')
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    test_data = WritingDataset(dataset_folder='sign_data', img_size=(300, 150), mode='test')
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    for x, data in enumerate(trainLoader, 0):
        images = data['ref_img']
        images1 = data['test_img']
        g_truth = data['ground_truth']
        a = 0



        