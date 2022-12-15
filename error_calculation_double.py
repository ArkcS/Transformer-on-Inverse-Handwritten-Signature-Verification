import cv2
import numpy as np
import tqdm
import os
import double_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import Model.Modelv2 as modelv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = double_dataset.DoubleDataSet(data_folder_eng='dataset_cedar', data_folder_dutch='sign_data', img_size=(300, 150), mode='test')
testLoader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)
net = modelv2.InverseTransformer().to(device)
state_dict = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'doubledataset31.ckpt'))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:] # remove `module.`
    new_state_dict[namekey] = v
# load params
net.load_state_dict(new_state_dict)

#net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yyctrue_model3out30.ckpt')))
net.eval()

true_ones = 0
false_ones = 0

predict_true_but_false = 0
predict_false_but_true = 0

count = 0
t = 0

choose = 0
chose = 0

def auc_cal(pred):

    for i in range(3):
        pred[:, i][pred[:, i] > 0.5] = 1
        pred[:, i][pred[:, i] <= 0.5] = 0
    pred = pred[:, 0] + pred[:, 1] + pred[:, 2]

    pred[pred < 2] = 0
    pred[pred >= 2] = 1
    return pred


with torch.no_grad():
    for data in tqdm.tqdm(testLoader):
        image1 = data['ref_img'].to(device)
        image2 = data['test_img'].to(device)
        gt = data['ground_truth'].numpy()
        output1, output2, output3 = net(image1, image2)
        predict = torch.cat([output1, output2, output3], dim=1)
        result = auc_cal(predict).cpu().numpy()
        count += len(gt)
        for each in gt:
            if each == 0:
                false_ones += 1
            else:
                true_ones += 1
        for i in range(0, len(result)):
            if result[i] != gt[i]:
                if gt[i] == 0:
                    predict_true_but_false += 1
                else:
                    predict_false_but_true += 1
                '''if choose == 0:
                    choose = 1
                    img1 = image1[i].cpu().numpy()
                    img1 = img1.transpose(2, 1, 0).astype(np.uint8)
                    cv2.imshow('img1', img1)
                    cv2.waitKey(0)
                    cv2.imwrite(os.path.join(PATH_Save, 'img_wrong_ref.png'), img1)
                    img2 = image2[i].cpu().numpy()
                    img2 = img2.transpose(2, 1, 0).astype(np.uint8)
                    cv2.imshow('img2', img2)
                    cv2.waitKey(0)
                    cv2.imwrite(os.path.join(PATH_Save, 'img_wrong_test.png'), img2)
                    print(result[i], gt[i])'''
            else:
                t += 1
                '''if chose == 0:
                    chose = 1
                    im1 = image1[i].cpu().numpy()
                    im1 = im1.transpose(2, 1, 0).astype(np.uint8)
                    cv2.imshow('img1', im1)
                    cv2.waitKey(0)
                    cv2.imwrite(os.path.join(PATH_Save, 'img_right_ref.png'), im1)
                    im2 = image2[i].cpu().numpy()
                    im2 = im2.transpose(2, 1, 0).astype(np.uint8)
                    cv2.imshow('img2', im2)
                    cv2.waitKey(0)
                    cv2.imwrite(os.path.join(PATH_Save, 'img_right_test.png'), im2)
                    print(result[i], gt[i])'''

far = predict_true_but_false / false_ones
frr = predict_false_but_true / true_ones
acc = t / count
print('t_o', true_ones)
print('f_o', false_ones)
print('t', t)
print('count', count)
print('far', far)
print('frr', frr)
print('acc', acc)