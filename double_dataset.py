import torch
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader,Dataset
from glob import glob
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


class DoubleDataSet(Dataset):
    def __init__(self, data_folder_eng, data_folder_dutch, img_size, mode='train'):
        self.mode = mode
        self.img_size = img_size
        self.pairs = []
        if self.mode == 'train':
            self.idx = range(1, 61)
            for i in self.idx:
                choices = []
                img_path = sorted(glob('{}/train/{:0>3d}/*'.format(data_folder_dutch, i)))
                img_path_f = sorted(glob('{}/train/{:0>3d}_forg/*'.format(data_folder_dutch, i)))

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

                        self.pairs.append((ref_o, test_o, 1, 'dutch'))
                    else:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path_f[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 0, 'dutch'))

            forgs = sorted(os.listdir(os.path.join(data_folder_eng, "full_forg")), key=sort_method)
            orgs = sorted(os.listdir(os.path.join(data_folder_eng, "full_org")), key=sort_method)
            forg_images, org_images = [], []
            for i in range(24 * 50):
                image_f = cv2.imread(os.path.join(os.path.join(data_folder_eng, "full_forg"), forgs[i]))
                image_f = cv2.resize(image_f, img_size).astype(np.float32).transpose(2, 1, 0)
                forg_images.append(image_f)
                image_t = cv2.imread(os.path.join(os.path.join(data_folder_eng, "full_org"), orgs[i]))
                image_t = cv2.resize(image_t, img_size).astype(np.float32).transpose(2, 1, 0)
                org_images.append(image_t)

            for i in range(50):  # for each one in 55 kinds of names
                temp_list = []
                for m in range(24):
                    for n in range(m+1, 24):
                        self.pairs.append((org_images[i * 24 + m], org_images[i * 24 + n], 1, 'english'))

                for j in range(24):
                    for k in range(24):
                        temp_list.append((org_images[i * 24 + j], forg_images[i * 24 + k], 0, 'english'))

                random.shuffle(temp_list)
                temp_list = temp_list[:276]
                self.pairs.extend(temp_list)

        elif self.mode == 'test':
            self.idx = range(61, 70)
            for i in self.idx:
                choices = []
                img_path = sorted(glob('{}/test/{:0>3d}/*'.format(data_folder_dutch, i)))
                img_path_f = sorted(glob('{}/test/{:0>3d}_forg/*'.format(data_folder_dutch, i)))

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

                        self.pairs.append((ref_o, test_o, 1, 'dutch'))
                    else:
                        ref_o = cv2.imread(img_path[each[0]])
                        ref_o = cv2.cvtColor(ref_o, cv2.COLOR_BGR2GRAY)
                        ref_o = np.stack((ref_o,) * 3, axis=2)
                        ref_o = cv2.resize(ref_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        test_o = cv2.imread(img_path_f[each[1]])
                        test_o = cv2.cvtColor(test_o, cv2.COLOR_BGR2GRAY)
                        test_o = np.stack((test_o,) * 3, axis=2)
                        test_o = cv2.resize(test_o, self.img_size).astype(np.float32).transpose(2, 1, 0)

                        self.pairs.append((ref_o, test_o, 0, 'dutch'))

            forgs = sorted(os.listdir(os.path.join(data_folder_eng, "full_forg")), key=sort_method)
            orgs = sorted(os.listdir(os.path.join(data_folder_eng, "full_org")), key=sort_method)
            forg_images, org_images = [], []
            for i in range(24*50, 24 * 55):
                image_f = cv2.imread(os.path.join(os.path.join(data_folder_eng, "full_forg"), forgs[i]))
                image_f = cv2.resize(image_f, img_size).astype(np.float32).transpose(2, 1, 0)
                forg_images.append(image_f)
                image_t = cv2.imread(os.path.join(os.path.join(data_folder_eng, "full_org"), orgs[i]))
                image_t = cv2.resize(image_t, img_size).astype(np.float32).transpose(2, 1, 0)
                org_images.append(image_t)

            for i in range(0, 5):  # for each one in 55 kinds of names
                temp_list = []
                for m in range(24):
                    for n in range(m + 1, 24):
                        self.pairs.append((org_images[i * 24 + m], org_images[i * 24 + n], 1, 'english'))

                for j in range(24):
                    for k in range(24):
                        temp_list.append((org_images[i * 24 + j], forg_images[i * 24 + k], 0, 'english'))

                random.shuffle(temp_list)
                temp_list = temp_list[:276]
                self.pairs.extend(temp_list)
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
        language = self.pairs[index][3]
        return {"img1": ref.float().to(device), "img2": test.float().to(device), 'gt': ground_truth.float().to(device), 'language': language}


if __name__ == '__main__':
    train_data = DoubleDataSet(data_folder_eng='dataset_cedar', data_folder_dutch='sign_data', img_size=(300, 150), mode='train')
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    test_data = DoubleDataSet(data_folder_eng='dataset_cedar', data_folder_dutch='sign_data', img_size=(300, 150), mode='test')
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    for x, data in enumerate(trainLoader, 0):
        images = data['ref_img']
        images1 = data['test_img']
        g_truth = data['ground_truth']
        l = data['language']
    for x, data in enumerate(testLoader, 0):
        images = data['ref_img']
        images1 = data['test_img']
        g_truth = data['ground_truth']
        l = data['language']