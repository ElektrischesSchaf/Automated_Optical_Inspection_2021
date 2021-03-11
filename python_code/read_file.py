import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from PIL import Image

import os

import pandas as pd

batch_size=64

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform=transforms.ToTensor()

def default_loader(path):
    return torchvision.dataset.ImageFolder(
        root =  path,
        transform = torchvision.transforms.ToTensor()
    )

# https://stackoverflow.com/questions/50052295/how-do-you-load-mnist-images-into-pytorch-dataloader
class MyTrainDataset(Dataset):
    def __init__(self, text_file,root_dir, transform=transform):
        self.csv_file=pd.read_csv(text_file)
        self.name_frame=self.csv_file['ID']
        self.label_frame=self.csv_file['Label']
        self.root_dir = root_dir
        self.transform = transform
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.name_frame.iloc[idx] ))
        labels = self.label_frame.iloc[idx]
        image = Image.open(img_name)
        image = self.transform(image)
        sample={'image':image, 'labels':labels}
        return sample

    def __len__(self):
        return len(self.name_frame)

class MyTestDataset(Dataset):
    def __init__(self, text_file,root_dir, transform=transform):
        self.csv_file=pd.read_csv(text_file)
        self.name_frame=self.csv_file['ID']

        self.root_dir = root_dir
        self.transform = transform
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.name_frame.iloc[idx] ))
        image = Image.open(img_name)
        image = self.transform(image)
        sample={'image':image}
        return sample

    def __len__(self):
        return len(self.name_frame)


train_dataset = MyTrainDataset( text_file= './dataset/train.csv' , root_dir='./dataset/train_images/train_images/')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = MyTestDataset( text_file= './dataset/test.csv' , root_dir='./dataset/test_images/test_images/')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(type(train_dataset))
print(type(test_dataset))

for batch_idx, data in enumerate(train_dataloader):
    print (data['labels'])
    # print('\n')
    # print (data['image'])


for batch_idx, data in enumerate(test_dataloader):
	print (data['image'].size())

'''
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

def load_train_dataset():
    train_data_path = './dataset/train_images/'


    train_dataset = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=torchvision.transforms.ToTensor()
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )    
    return train_loader

def load_test_dataset():
    test_data_path = './dataset/test_images/'

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform=torchvision.transforms.ToTensor()
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )

    return test_loader

# load_dataset()

for batch_idx, (data, target) in enumerate(load_train_dataset()):
	print (batch_idx)

for batch_idx, (data, target) in enumerate(load_test_dataset()):
	print (batch_idx)
'''