# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:40:05 2021

@author: li xiang
"""
import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FaceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        
        self.label_set = pd.read_csv(csv_file)
        self.index_set = pd.read_csv('/store/pa2/index.txt', sep=",", header=None)
        self.transform = transform

    def __getitem__(self, index = None):
        if index == None:
            i = np.random.randint(0,len(self.label_set)-1)
        else:
            i = index
        id1 = self.label_set['id1'][i]
        id2 = self.label_set['id2'][i]
        target = np.array([self.label_set['target'][i]]).astype(np.float32)
        Image1 = Image.open('/store/pa2/'+self.index_set[1][int(id1)-1]).convert('L')
        Image2 = Image.open('/store/pa2/'+self.index_set[1][int(id2)-1]).convert('L')
        Image1 = self.transform(Image1)
        Image2 = self.transform(Image2)
        return Image1, Image2, target
    def __len__(self):
        return len(self.label_set)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),  
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=3),  
        nn.BatchNorm2d(32), 
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3), 
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3), 
        nn.BatchNorm2d(128), 
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3), 
        nn.BatchNorm2d(256), 
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3),
        nn.BatchNorm2d(512), 
        nn.ReLU(),
        nn.AvgPool2d(6,1))
        
        self.fc = nn.Sequential(
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512,1),
        nn.Sigmoid()
        )

    def forward(self, x1, x2):
        #1,32,32
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        x = torch.abs(x1 - x2)
        x = x.view(-1,1,512)
        x = self.fc(x)
        x = x.view(-1,1)
        return x
    

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
from tqdm.notebook import tqdm
# pip install tqdm

def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)
    print('Model saved to {save_path}')

def load_checkpoint(save_path, model, optimizer):
    save_path = save_path #f'cifar_net.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print('Model loaded from {save_path}, with val loss: {val_loss}')
    return val_loss



from tqdm.notebook import tqdm
# pip install tqdm

def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)
    print('Model saved to {save_path}')

def load_checkpoint(save_path, model, optimizer):
    save_path = save_path #f'cifar_net.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print('Model loaded from {save_path}, with val loss: {val_loss}')
    return val_loss



def TRAIN(net, train_loader, valid_loader,  num_epochs, criterion, optimizer, val_loss, device, save_name):
    
    if val_loss==None:
        best_val_loss = float("Inf")  
    else: 
        best_val_loss=val_loss
        print('Resume training')

    training_step = 0
    training_loss = []
    validation_loss = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs1, inputs2, labels in tqdm(train_loader):
            
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            '''Training of the model'''
            # Forward pass
            outputs = net(inputs1,inputs2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1
            if training_step%10 == 0:
                training_loss.append(loss.item())
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            with torch.no_grad():
                net.eval()
                running_val_loss = 0.0
                running_val_corrects = 0
                for inputs1_valid, inputs2_valid, labels_valid in (valid_loader):
                    inputs1_valid = inputs1_valid.to(device)
                    inputs2_valid = inputs2_valid.to(device)
                    labels_valid = labels_valid.to(device)

                    outputs = net(inputs1_valid, inputs2_valid)
                    loss = criterion(outputs, labels_valid)
                    
                    running_val_loss += loss.item()
                    _, preds = torch.max(outputs.data, 1)
                    running_val_corrects += torch.sum(preds == labels_valid.data)
                if training_step%10 == 0:
                    validation_loss.append(running_val_loss / len(valid_loader))

        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects / float(len(train_loader.dataset))
        
        with torch.no_grad():
            net.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs1_valid, inputs2_valid, labels_valid in tqdm(valid_loader):
                inputs1_valid = inputs1_valid.to(device)
                inputs2_valid = inputs2_valid.to(device)
                labels_valid = labels_valid.to(device)

                outputs = net(inputs1_valid, inputs2_valid)
                loss = criterion(outputs, labels_valid)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels_valid.data)

            valid_loss = running_loss / len(valid_loader)
            valid_acc = running_corrects / float(len(valid_loader.dataset))
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f},  Valid Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_checkpoint(save_name, net, optimizer, best_val_loss)
        
    print('Finished Training')
    return training_loss, validation_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    model = Net()
    transform = transforms.Compose(
    [transforms.Resize([32,32]),
     transforms.RandomHorizontalFlip(p=0.5), # data augmentation by fliping 
     transforms.ToTensor(),    # range [0, 255]  -> [0.0,1.0] Convert a PIL Image or numpy.ndarray (H x W x C)  to tensor (C x H x W) 
     transforms.Normalize((0.5,), (0.5,))   # channel=（channel-mean）/std  -> [-1, 1]
     ])
    train_set = FaceDataset('train.csv',transform)
    valid_set = FaceDataset('valid.csv',transform)
    test_set = FaceDataset('test.csv',transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True)
    pp = get_n_params(model)
    print(pp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    best_val_loss = None
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    save_path = 'cifar_net.pt'
    model = model.to(device)
    training_loss, validation_loss = TRAIN(model, train_loader, valid_loader, num_epochs, criterion, optimizer, best_val_loss, device, save_path)
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.show()
    total_label = []
    total_output = []
    for inputs1_valid, inputs2_valid, labels_valid in tqdm(valid_loader):
        inputs1_valid = inputs1_valid.to(device)
        inputs2_valid = inputs2_valid.to(device)
        labels_valid = labels_valid.to(device)
        
        outputs = model(inputs1_valid, inputs2_valid)
        total_label.extend(labels_valid.detach().numpy())
        total_output.extend(outputs.detach().numpy())
    fpr,tpr,thresholds=roc_curve(total_label,
                                  total_output,
                                  pos_label=None,
                                  sample_weight=None,
                                  drop_intermediate=True)

    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label="siamese, area=%0.2f)" % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([0.00, 1.0])
    plt.ylim([0.00, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")



