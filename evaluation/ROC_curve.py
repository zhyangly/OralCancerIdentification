# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:34:15 2021
draw ROC curve

@author: yangzihan
"""



import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
import argparse
import torchvision.models as models
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interp

from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='F:/yzh/data_process_results/DL/model/', help='folder to output images and model checkpoints') #模型保存路径
parser.add_argument('--net', default='/model/net.pth', help="path to netG (to continue training)")  
opt = parser.parse_args()
num_class = 3   # 类别数量

# EPOCH = 30   #遍历数据集次数
BATCH_SIZE = 32     #批处理尺寸(batch_size)
LR = 0.0001        #学习率
# root = 'D:/neck_tissue_datasets/'
dataset = 'F:/yzh/DeepLearningDataSet/'
# dataset = 'D:/NeckTissue/dataset1103/'
savepath = '/model/'

def default_loader(path):
    bgr = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb).convert('L')

    
    # return Image.open(path).convert('L')    
    return img            #定义加载图的模式
class MyDataset(Dataset):               #创建一个读图的类，这个类继承了torch.utils.data里的Dataset类
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')                 #打开指定的文件夹
        imgs = []                                   
        for line in fh:
            line = line.strip('\n')  #去掉换行符
            line = line.rstrip()     #去掉前后的空格
            words = line.split()     #根据中间的空格将words分段
            imgs.append((words[0],int(words[1]))) #将地址和标签添加到list
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
#        print(img.size())
        return img,label

    def __len__(self):
        return len(self.imgs)


#读取数据
transform1 = transforms.Compose([
                                 transforms.ToTensor(),
                                 
                                 # transforms.Grayscale(1)
                                 # transforms.Normalize((0.5,),(0.5,))
                                 ]
                                )
transform2 = transforms.ToTensor()
train_data1=MyDataset(txt=dataset+'train1.txt', transform=transform1)
train_data2=MyDataset(txt=dataset+'train2.txt', transform=transform1)
train_data3=MyDataset(txt=dataset+'train3.txt', transform=transform1)
train_data4=MyDataset(txt=dataset+'train4.txt', transform=transform1)
test_data1=MyDataset(txt=dataset+'test1.txt', transform=transform1)
test_data2=MyDataset(txt=dataset+'test2.txt', transform=transform1)
test_data3=MyDataset(txt=dataset+'test3.txt', transform=transform1)
test_data4=MyDataset(txt=dataset+'test4.txt', transform=transform1)
train_data=train_data1+test_data1+train_data2+test_data2+train_data3+test_data3
test_data=train_data4+test_data4
# train_data=MyDataset(txt=dataset+'train1.txt', transform=transform1)
# test_data=MyDataset(txt=dataset+'test1.txt', transform=transform1)

#加载数据
# trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#testloader = DataLoader(dataset=test_data, batch_size=8)
testloader = DataLoader(dataset=test_data)
train_data_size = 40500
valid_data_size = 13500

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 29 * 29, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 5)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
net = torch.load('F:/yzh/DeepLearningDataSet/model1/ResNet1840.pt')
# net = LeNet()

# net.load_state_dict(torch.load("F:/phm1116/lenet/_history.pt"))




print(net)
net = net.to(device)

loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
# loss_func = nn.NLLLoss()
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# optimizer = optim.Adam(net.parameters(),lr = LR)


def train_and_valid(model, loss_function):

    valid_loss = 0.0

    valid_acc = 0.0
    valid_acc2 = 0.0
    valid_acc3 = 0.0
    valid_acc4 = 0.0
    # valid_acc5 = 0.0
    # valid_acc6 = 0.0

    _0_acc4 = 0
    _1_acc4 = 0
    # _3_acc4 = 0
    # _4_acc4 = 0
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签
    with torch.no_grad():

        model.eval()



        for j, (inputs, labels) in enumerate(testloader):

            inputs = inputs.to(device)

            labels = labels.to(device)

                

            outputs = model(inputs)
            # print(outputs)
            score_tmp = outputs  # (batchsize, nclass)
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            
       
            
            loss = loss_function(outputs, labels)

 

            valid_loss += loss.item() * inputs.size(0)

 

            ret, predictions = torch.max(outputs.data, 1)
            # print(ret)
            # print(predictions)

            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # print(correct_counts)

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # print(acc)

            valid_acc += acc.item() * inputs.size(0)
            # print(valid_acc)
            
            if labels.data == 0:
                correct_counts2 = predictions.eq(labels.data.view_as(predictions))
                acc2 = torch.mean(correct_counts2.type(torch.FloatTensor))
                valid_acc2 += acc2.item() * inputs.size(0)
            if labels.data == 1:
                correct_counts3 = predictions.eq(labels.data.view_as(predictions))
                acc3 = torch.mean(correct_counts3.type(torch.FloatTensor))
                valid_acc3 += acc3.item() * inputs.size(0)
            if labels.data == 2:
                # a = tensor(0)
                _0_counts4 = predictions.eq(torch.tensor(0))
                _0acc4 = torch.mean(_0_counts4.type(torch.FloatTensor))
                _0_acc4 += _0acc4.item() * inputs.size(0)
                
                _1_counts4 = predictions.eq(torch.tensor(1))
                _1acc4 = torch.mean(_1_counts4.type(torch.FloatTensor))
                _1_acc4 += _1acc4.item() * inputs.size(0)      
                
                correct_counts4 = predictions.eq(labels.data.view_as(predictions))
                acc4 = torch.mean(correct_counts4.type(torch.FloatTensor))
                valid_acc4 += acc4.item() * inputs.size(0) 
                
                # _3_counts4 = predictions.eq(torch.tensor(3))
                # _3acc4 = torch.mean(_3_counts4.type(torch.FloatTensor))
                # _3_acc4 += _3acc4.item() * inputs.size(0)    
                
                # _4_counts4 = predictions.eq(torch.tensor(4))
                # _4acc4 = torch.mean(_4_counts4.type(torch.FloatTensor))
                # _4_acc4 += _4acc4.item() * inputs.size(0)
                
            # if labels.data == 3:
            #     correct_counts5 = predictions.eq(labels.data.view_as(predictions))
            #     acc5 = torch.mean(correct_counts5.type(torch.FloatTensor))
            #     valid_acc5 += acc5.item() * inputs.size(0)
            # if labels.data == 4:
            #     correct_counts6 = predictions.eq(labels.data.view_as(predictions))
            #     acc6 = torch.mean(correct_counts6.type(torch.FloatTensor))
            #     valid_acc6 += acc6.item() * inputs.size(0)
        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
     
        print("score_array:", score_array.shape)  # (batchsize, classnum)
        print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
     
        # 调用sklearn库，计算每个类别对应的fpr和tpr
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(num_class):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
     
        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= num_class
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
        # print(fpr_dict["micro"])
        # print(tpr_dict["micro"])
        # np.savetxt(r'F:\yzh\data_process_results\DL\Lenet_fpr.txt',
        #     fpr_dict, fmt='%.5f', delimiter=',')
        # np.savetxt(r'F:\yzh\data_process_results\DL\Lenet_tpr.txt',
        #     tpr_dict, fmt='%.5f', delimiter=',')
        # 绘制所有类别平均的roc曲线
        plt.figure()
        lw = 2
        plt.plot(fpr_dict["micro"], tpr_dict["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
     
        # plt.plot(fpr_dict["macro"], tpr_dict["macro"],
        #          label='macro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc_dict["macro"]),
        #          color='navy', linestyle=':', linewidth=4)
     
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(num_class), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc_dict[i]))
       
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc="lower right")
        
        # plt.savefig('Resnet18_roc.tiff',dpi=600)
        plt.show()
    print("预测类1的数量",valid_acc2,"类1的正确率",(valid_acc2/1601))  
    print("预测类2的数量",valid_acc3,"类2的正确率",(valid_acc3/443))  
    print("预测类3的数量",valid_acc4,"类3的正确率",(valid_acc4/1418),'各类正确个数',_0_acc4,_1_acc4,valid_acc4) 
    # print("预测类4的数量",valid_acc5,"类4的正确率",(valid_acc5/3000))  
    # print("预测类5的数量",valid_acc6,"类5的正确率",(valid_acc6/3000))            
                
                
                
    avg_valid_loss = valid_loss/valid_data_size

    avg_valid_acc = valid_acc/valid_data_size




 

    print("tValidation: Loss: {:.4f}, Accuracy: {:.4f}%".format(

        avg_valid_loss, avg_valid_acc*100

    ))

    
    return model,fpr_dict,tpr_dict

# num_epochs = 30
    
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
trained_model, test_fpr_dict, test_tpr_dict = train_and_valid(net, loss_func)

# torch.save(history, dataset + savepath + '_history.pt')

 

# history = np.array(history)

# plt.plot(history[:, 0:2])

# plt.legend(['Tr Loss', 'Val Loss'])

# plt.xlabel('Epoch Number')

# plt.ylabel('Loss')

# plt.ylim(0, 1)

# plt.savefig(dataset + savepath + '_loss_curve.png')

# plt.show()

 

# plt.plot(history[:, 2:4])

# plt.legend(['Tr Accuracy', 'Val Accuracy'])

# plt.xlabel('Epoch Number')

# plt.ylabel('Accuracy')

# plt.ylim(0, 1)

# plt.savefig(dataset + savepath + '_accuracy_curve.png')

# plt.show()


#D = pd.DataFrame(test_fpr_dict[2]).applymap(lambda x:('%.5f')%x) #将dict的数据保存出来











































