import cv2
import numpy as np
import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import Model.Modelv2 as modelv2
import dataset_wyc_backup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net = modelv2.InverseTransformer().to(device)

state_dict = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yyctrue_model3out39.ckpt'))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:] # remove `module.`
    new_state_dict[namekey] = v
# load params
net.load_state_dict(new_state_dict)
#net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yyctrue_model3out39.ckpt')))

test_data = dataset_wyc_backup.dataset("dataset_cedar", (300, 150), 'test')
testLoader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)

net.eval()

true_ones = 0
false_ones = 0

predict_true_but_false = 0
predict_false_but_true = 0

count = 0
t = 0


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
        image1 = data['img1'].to(device)
        image2 = data['img2'].to(device)
        gt = data['gt'].cpu().numpy()
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
            else:
                t += 1

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
