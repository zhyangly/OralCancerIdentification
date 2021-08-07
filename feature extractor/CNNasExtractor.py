# -*- coding: utf-8 -*-
"""
Edited by Zhyang 
nn as a extractor, output a feature vector 

"""

import numpy as np

import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = 'F:/yzh/data_process_results/DL/PreprocessDataset/'
# dataset = 'D:/NeckTissue/dataset1103/'
# savepath = '/model/'
# modelname = 'resnet18'

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

BATCH_SIZE = 32     #批处理尺寸(batch_size)
#读取数据
transform1 = transforms.Compose([
                                 transforms.ToTensor(),
                                 
                                 # transforms.Grayscale(1)
                                 # transforms.Normalize((0.5,),(0.5,))
                                 ]
                                )
transform2 = transforms.ToTensor()
train_data=MyDataset(txt=dataset+'train1.txt', transform=transform1)
test_data=MyDataset(txt=dataset+'test1.txt', transform=transform1)

#加载数据
trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#testloader = DataLoader(dataset=test_data, batch_size=8)
testloader = DataLoader(dataset=test_data)

train_data_size = 9738  #总的训练数据大小（包含所有类）
valid_data_size = 3462  #总的测试集大小（包含所有类）

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
        self.fc3 = nn.Linear(84, 3)

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

net = torch.load('F:/yzh/data_process_results/DL/PreprocessDataset/model/VGG16_240.pt') 
#net.fc = nn.Sequential()

# print(net) 
#model = torchvision.models.vgg16(pretrained=True)
#实际实用的时候，a可以换成我们需要输出的图片
#a=trainloader #a=torch.zeros((2,3,224,224))
#我们枚举整个网络的所有层
b=np.zeros(shape=(1,4096))  #一个batchsize的特征维度，依不同神经网络而定
d=np.zeros(shape=(1),dtype=np.int64) # 标签的特征维度
k=0
epoch_start = time.time()
for i, (inputs, labels) in enumerate(trainloader):

    inputs = inputs.to(device)
    # print(inputs.size())
    labels = labels.to(device)

    for i,m in enumerate(net.modules()):
     	#让网络依次经过和原先结构相同的层，我们就可以获取想要的层的输出
     	if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or\
     			isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
    		# print(m)
            
            inputs= m(inputs)
            # print(inputs.size())
     	#我只想要第一个全连接层的输出
     	elif isinstance(m, nn.Linear):
            #print(m)
    		#和源代码一样，将他展平成一维的向量
            inputs = torch.flatten(inputs, 1)
    		#获取第一个全连接层的输出
            inputs= m(inputs)
            # print(inputs.size())
            break
    
    #inputs=net(inputs)
    
    k=k+1
    print(k)
    a = inputs.detach().numpy()
    b=np.vstack((b,a))
    c = labels.numpy()
    
    d = np.hstack((d,c))
epoch_end = time.time()
e = epoch_end-epoch_start
# .ExcelWriter('hhhh.xlsx')  #关键2，创建名称为hhh的excel表格
# data_df.to_edata_df = pd.DataFrame(b)
# writer = pdxcel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# writer.save()  #关键4                                                                                                  