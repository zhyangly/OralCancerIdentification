# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:56:33 2020

@author: Administrator
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='F:/yzh/DeepLearningDataSet/model_16/', help='folder to output images and model checkpoints') 
parser.add_argument('--net', default='/model/net.pth', help="path to netG (to continue training)")  
opt = parser.parse_args()


# EPOCH = 30   
BATCH_SIZE = 32     #(batch_size)
LR = 0.001        #learning rate
# root = 'D:/neck_tissue_datasets/'
dataset = 'F:/yzh/DeepLearningDataSet/'
# dataset = 'D:/NeckTissue/dataset1103/'
savepath = '/model_16/'
modelname = '1'

def default_loader(path):
    bgr = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb).convert('L')

    
    # return Image.open(path).convert('L')    
    return img            
class MyDataset(Dataset):              
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')                 #Open the specified folder
        imgs = []                                   
        for line in fh:
            line = line.strip('\n')  
            line = line.rstrip()     
            words = line.split()     
            imgs.append((words[0],int(words[1]))) 
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


#Reading data
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

train_data=train_data2+test_data2+train_data3+test_data3+train_data4+test_data4
test_data=train_data1+test_data1

trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#testloader = DataLoader(dataset=test_data, batch_size=8)
testloader = DataLoader(dataset=test_data)

train_data_size = 40500  #Total training set (including all classes)
valid_data_size = 13500  #Total test set (including all classes)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), 
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

   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # nn.Linear()
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class LeNet12(nn.Module):
    def __init__(self):
        super(LeNet12, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), 
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),      #input_size=(16*10*10)
            # nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 3)

    
    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = self.conv5(x)
        # print(x.size())
        x = self.conv6(x)
        # print(x.size())
        # nn.Linear()
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
# net = LeNet12().to(device)
# net = LeNet()
# net = torch.load('F:/yzh/data_process_results/DL/PreprocessDataset/model/Lenet_240.pt') #Load the trained model
# resnet50 = models.resnet18(pretrained=True)
# net = models.squeezenet1_0(num_classes = 5)
# print(net)

#resnet18
# net = models.resnet18(pretrained=False)
# net = models.resnet18(num_classes = 3)
# net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  #Change layer 1 of resnet18 to single channel input
# net.features[0] = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
# for param in resnet50.parameters():

# VGG16

net = models.vgg16(pretrained=False)    
# net = net.vgg16(num_classes=3) 
net.features[0]=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #Change layer 1 of vgg16 to single channel input
net.classifier[6]=nn.Linear(in_features=4096,out_features=3,bias=True) #Modify the output of the final full connection layer
#     param.requires_grad = False


# fc_inputs = net.fc.in_features

# net.fc = nn.Sequential(

#     nn.Linear(fc_inputs, 256),

#     nn.ReLU(),

#     nn.Dropout(0.4),

#     nn.Linear(256, 5),

#     nn.LogSoftmax(dim=1)

# )

# net.classifier = nn.Sequential(
#     nn.Dropout(p=0.5, inplace=False),

#     nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1)),

#     nn.ReLU(),


#     nn.LogSoftmax(dim=1)

# )
print(net)
net = net.to(device)

loss_func = nn.CrossEntropyLoss()  
# loss_func = nn.NLLLoss()
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(net.parameters(),lr = LR)

def train_and_valid(model, loss_function, optimizer, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = []

    best_acc = 0.0

    best_epoch = 0

 

    for epoch in range(epochs):

        epoch_start = time.time()

        print("Epoch: {}/{}".format(epoch+1, epochs))

 

        model.train()

 

        train_loss = 0.0

        train_acc = 0.0

        valid_loss = 0.0

        valid_acc = 0.0

 

        for i, (inputs, labels) in enumerate(trainloader):

            inputs = inputs.to(device)

            labels = labels.to(device)

 

           

            optimizer.zero_grad()

 

            outputs = model(inputs)

 

            loss = loss_function(outputs, labels)

 

            loss.backward()

 

            optimizer.step()

 

            train_loss += loss.item() * inputs.size(0)

 

            ret, predictions = torch.max(outputs.data, 1)

            correct_counts = predictions.eq(labels.data.view_as(predictions))

 

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

 

            train_acc += acc.item() * inputs.size(0)

 

        with torch.no_grad():

            model.eval()

 

            for j, (inputs, labels) in enumerate(testloader):

                inputs = inputs.to(device)

                labels = labels.to(device)

 

                outputs = model(inputs)

 

                loss = loss_function(outputs, labels)

 

                valid_loss += loss.item() * inputs.size(0)

 

                ret, predictions = torch.max(outputs.data, 1)

                correct_counts = predictions.eq(labels.data.view_as(predictions))

 

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

 

                valid_acc += acc.item() * inputs.size(0)

 

        avg_train_loss = train_loss/train_data_size

        avg_train_acc = train_acc/train_data_size

 

        avg_valid_loss = valid_loss/valid_data_size

        avg_valid_acc = valid_acc/valid_data_size

 

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

 

        if best_acc < avg_valid_acc:

            best_acc = avg_valid_acc

            best_epoch = epoch + 1

 

        epoch_end = time.time()

 

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(

            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start

        ))

        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

 

        torch.save(model, dataset + savepath + modelname+str(epoch+1)+'.pt')
    

    return model, history

num_epochs = 40

trained_model, history = train_and_valid(net, loss_func, optimizer, num_epochs)

# torch.save(history, dataset + savepath + modelname+'_history.pt')

 

history = np.array(history)

plt.plot(history[:, 0:2])

plt.legend(['Tr Loss', 'Val Loss'])

plt.xlabel('Epoch Number')

plt.ylabel('Loss')

plt.ylim(0, 1)

# plt.savefig(dataset + savepath + modelname+'_loss_curve.png')

plt.show()

 

plt.plot(history[:, 2:4])

plt.legend(['Tr Accuracy', 'Val Accuracy'])

plt.xlabel('Epoch Number')

plt.ylabel('Accuracy')

plt.ylim(0, 1)

# plt.savefig(dataset + savepath + modelname+'_accuracy_curve.png')

plt.show()






























