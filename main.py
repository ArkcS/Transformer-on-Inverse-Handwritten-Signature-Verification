import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import torch
from Model.Modelv3 import InverseTransformer as modelnet
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import os
from numpy import *
from PIL import Image
from data_set_fjnew import ImageLoader
import visdom
import Dataset_sign_data
import random
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossFunc:
    def __init__(self):
        super(LossFunc, self).__init__()
        self.bce_loss = nn.MSELoss()

    def lossfc(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.4, 0.3, 0.3
        # print(max(x), max(label))
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        return alpha_1 * loss_1 + alpha_2 * loss_2 + alpha_3 * loss_3


class Visdom_draw(object):
    def __init__(self, env_name='main'):
        self.visdom = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.visdom.line(X=array([x, x]), Y=array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.visdom.line(X=array([x]), Y=array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                             update='append')


def auc_cal(pred, label):

    for i in range(3):
        pred[:, i][pred[:, i] > 0.5] = 1
        pred[:, i][pred[:, i] <= 0.5] = 0
    pred = pred[:, 0] + pred[:, 1] + pred[:, 2]

    pred[pred < 2] = 0
    pred[pred >= 2] = 1
    pred = pred.view(-1)

    label = label.view(-1)

    auc = torch.sum(pred == label).item() / label.size()[0]
    
    return auc


if __name__ == '__main__':
    model = modelnet().cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    plotter = Visdom_draw(env_name="fjtrain")
    # x = torch.randn(2,3,600,300)
    # y = torch.randn(2, 3, 600, 300)
    # out = net(x,y)

    # train_data=DataLoader(Dataset_sign_data.WritingDataset('sign_data',(600,300)))

    # wyc dataset
    train_pair = []
    test_pair = []
    PATH = r"/remote-home/yyc/VisionEMG/cs172_final/DATA/signatures"

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

    trainData = ImageLoader(os.path.join(PATH), load_type="train", transform=None, choices=train_pair)
    testData = ImageLoader(os.path.join(PATH), load_type="test", transform=None, choices=test_pair)



    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # curr_lr = learning_rate
    loss_fc = LossFunc()

    # wind = Visdom()
    # wind.line([],
    #           [],
    #           win='train_loss',
    #           opts=dict(title='train_loss')
    #           )

    epoches = 300
    trainsampler = [i for i in range(len(trainData))]
    testsampler = [i for i in range(len(testData))]
    for epoch in range(epoches):
        loss_list = []
        auc_list = []
        model.train()
        
        random.shuffle(trainsampler)
        trainLoader = torch.utils.data.DataLoader(trainData,
                                             batch_size=128,
                                             sampler=trainsampler)
        random.shuffle(testsampler)
        testLoader = torch.utils.data.DataLoader(testData,
                                             batch_size=128,
                                             sampler=testsampler)
        for data in tqdm.tqdm(trainLoader):
            """"
            image1=data[0].to(torch.float32)
            image2=data[1].to(torch.float32)
            label=data[2].to(torch.float32)
            """

            image1 = data['ref_img']
            image2 = data['test_img']
            label = data['ground_truth']

            # image1 = image1.to(device)
            # image2 = image2.to(device)
            # label = label.to(device)
            optimizer.zero_grad()
            output1, output2, output3 = model(image1, image2)

            # outputs = outputs.squeeze()

            # print(outputs.size())
            # print(depth.size())

            loss = loss_fc.lossfc(output1, output2, output3, label)

            loss.backward()
            optimizer.step()

            loss_list.append(float(loss.detach()))

        

            pred = torch.cat([output1, output2, output3], dim=1)
            auc_list.append(auc_cal(pred, label))
            # print("auc",auc_cal(pred, label))
            print("loss",loss.tolist())
            # wind.line([float(loss)],[i + 1],win = 'train_loss',update = 'append')

            # if (i + 1) % 50 == 0:
            #     print("Epoch [{}/{}], Step [{}/{}]"
            #           .format(epoch + 1, epoches, i + 1, total_step))
            # print(float(loss.detach()))
        plotter.plot('loss', 'train', 'total Loss', epoch, average(loss_list))
        plotter.plot('auc', 'train', 'auc', epoch, average(auc_list))
        auc_list = []
        with torch.no_grad():
            model.eval()
            loss_list = []
            for data in tqdm.tqdm(testLoader):
                image1 = data['ref_img']
                image2 = data['test_img']
                label = data['ground_truth']
                output1, output2, output3 = model(image1, image2)
                loss = loss_fc.lossfc(output1, output2, output3, label)
                loss_list.append(float(loss.detach()))
                pred = torch.cat([output1, output2, output3], dim=1)
                auc_list.append(auc_cal(pred, label))
                # print("auc",auc_cal(pred, label))
                print("loss",loss.tolist())
        plotter.plot('auc', 'vaild', 'auc', epoch, average(auc_list))
        plotter.plot('loss', 'vaild', 'total Loss', epoch, average(loss_list))

        torch.save(model.state_dict(), "model3out" + str(epoch) + ".ckpt")
        print("success saved")

    # torch.save(model.state_dict(), 'model.ckpt')



