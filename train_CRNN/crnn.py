import numpy as np
import torch

PATH_images = "./cropped/"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LETTER_BOX = ["BLANK", '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
 ' ', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '_',
 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',]

import json, os
_,_,image_list = next(os.walk(PATH_images))
image_list.sort()
image_list_train = image_list[:1160000]
image_list_test = image_list[1160000:]

from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import PILToTensor
from torch.optim import RMSprop
from PIL import Image
from torch import nn

class DatasetWDT(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.length = len(self.image_list)
        self.pil2tensor = PILToTensor()
    def __len__(self):
        return self.length
    def __getitem__(self, item):
        img, label = self._getitem(item)
        listt = []
        listt.append((img,label))
        width, height = img.size
        while len(listt)<BATCH_SIZE:
            item = item+1 if item+1<self.length else 0
            img, label = self._getitem(item)
            if width-16<img.size[0]<width+16:
                img = img.resize((width, height))
                listt.append((img,label))
        imgs = torch.cat([self.pil2tensor(img)[None] for img,_ in listt], dim=0)/512-0.25
        targets = torch.cat([self._makelabel(label) for _,label in listt], dim=0)
        target_lengths = torch.tensor([len(i[1]) for i in listt])
        # imgs = (self.pil2tensor(img)/512-0.25)[None]
        # targets = self._makelabel(label)
        # target_lengths = torch.tensor(len(targets))
        return imgs, targets, target_lengths
    def _getitem(self, item):
        img = Image.open(PATH_images+self.image_list[item])
        img = self._resize(img)
        label = self.image_list[item].split("_")[-2]
        return img, label
    def _resize(self, img, height=32):
        width = int(img.size[0]/img.size[1]*height+0.5)
        img = img.resize((width,height))
        return img
    def _makelabel(self, label):
        numbers = [LETTER_BOX.index(i) for i in label]
        numbers = torch.tensor(numbers)
        return numbers

dataset = DatasetWDT(image_list_train)
dataloader = DataLoader(dataset, num_workers=0, shuffle=True, collate_fn=lambda x:x[0])
dataset_test = DatasetWDT(image_list_test)
dataloader_test = DataLoader(dataset_test, num_workers=0, shuffle=False, collate_fn=lambda x:x[0])

import torch.nn as nn
#import params
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()
        def convRelu(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6)  # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)
        return output
    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero

model = CRNN(32,3,75,256).to(DEVICE)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
optimizer = RMSprop(params=model.parameters(), lr=0.00001)

if __name__=="__main__":
    for epoch in range(50):
        for i,(imgs,targets,target_lengths) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            imgs, targets, target_lengths = imgs.to(DEVICE), targets.to(DEVICE), target_lengths.to(DEVICE)
            output = model(imgs).log_softmax(2).requires_grad_()
            input_lengths = torch.tensor([output.shape[0]]*output.shape[1])
            loss = ctc_loss(output, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            if i%1000==0:
                print(f"trainloss:{loss.item()}")
            if i%100000==99999:
                torch.save(model,f"./train_result/model_{epoch}_{i}.pt")
                torch.save(optimizer, f"./train_result/model_{epoch}_{i}.optimizer")
                print("\tevaluating...")
                model.eval()
                losss = 0
                for j, (imgs, targets, target_lengths) in enumerate(dataloader_test):
                    model.eval()
                    imgs, targets, target_lengths = imgs.to(DEVICE), targets.to(DEVICE), target_lengths.to(DEVICE)
                    output = model(imgs).log_softmax(2)
                    input_lengths = torch.tensor([output.shape[0]] * output.shape[1])
                    loss = ctc_loss(output, targets, input_lengths, target_lengths)
                    losss += loss.item()
                print(f"evalloss:{losss/j}")

