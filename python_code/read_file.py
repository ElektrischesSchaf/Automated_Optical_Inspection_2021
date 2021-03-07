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