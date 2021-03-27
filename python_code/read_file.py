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

from tqdm import tqdm_notebook as tqdm
from tqdm import trange

learning_rate = 1e-5
max_epoch = 3
batch_size = 16

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
        if labels==0:
            out_labels=torch.tensor([0,0,0,0,0,0])
        if labels==1:
            out_labels=torch.tensor([1,0,0,0,0,0])
        if labels==2:
            out_labels=torch.tensor([0,1,0,0,0,0])
        if labels==3:
            out_labels=torch.tensor([0,0,1,0,0,0])
        if labels==4:
            out_labels=torch.tensor([0,0,0,1,0,0])
        if labels==5:
            out_labels=torch.tensor([0,0,0,0,1,0])
        if labels==6:
            out_labels=torch.tensor([0,0,0,0,0,1])

        image = Image.open(img_name)
        image = self.transform(image)
        # sample = {'image':image, 'labels':labels}
        return image, out_labels

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



class CNN_Simple(nn.Module):
    def __init__(self, batch_size,  in_channels, out_channels, kernel_heights, stride, padding):
        super(CNN_Simple, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d( in_channels, out_channels, (kernel_heights[0], kernel_heights[0]), self.stride, padding = self.padding)
        # self.conv2 = nn.Conv2d( in_channels, out_channels, (kernel_heights[1], kernel_heights[1]), stride, padding=0)

        self.l1 = nn.Linear(len(kernel_heights)*out_channels, int( len(kernel_heights)*out_channels/2) )

        self.linear = nn.Linear( 1*254*254, 6 )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        # print('conv_out.size()= ', conv_out.size(), '\n')
        # conv_out.size() = (batch_size, out_channels, dim, 1) 

        activation = F.relu(conv_out.squeeze(3))

        activation = activation.view(activation.size(0), -1)

        return activation
    def forward(self, x):

        # print(x.size())

        out = self.conv_block(x, self.conv1)

        out = self.linear(out)

        out = out.squeeze(1)

        # print('in forward: ', out.size(), '\n')

        return out


#  batch_size,  in_channels, out_channels, kernel_heights, stride, padding, 
model = CNN_Simple(batch_size, 3, 1, [3], 1, 0)


opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criteria = torch.nn.BCEWithLogitsLoss() # BCELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
history = {'train':[],'valid':[]}


class mean_recall():
    def __init__(self):
        self.n_predictions=0
        self.n_classes=0
        self.n_correct_predictions=0
        self.n_recall=0
        self.n_corrects=0
    def rest(self):
        self.n_predictions=0
        self.n_classes=0
        self.n_correct_predictions=0
    def update(self, predicts, ground_truth):
        self.n_recall += torch.sum(ground_truth).data.item()
        self.n_corrects += torch.sum(ground_truth.type(torch.bool)*predicts).data.item()
    def get_score(self):
        recall = self.n_corrects / self.n_recall
        return recall
    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)


def _run_epoch(epoch, mode):
    model.train(True)
    if mode=='train':
        description = 'train'
    trange = tqdm(enumerate( train_dataloader ), total=len(train_dataloader), desc=description )

    loss = 0
    MeanRecallScore = mean_recall()

    for i, (x, y) in trange:
        # print('_run_epoch type(x) ', type(x), ' x= ', x, '\n')
        # print('_run_epoch type(y。) ', type(y), ' y= ', y, '\n')
        o_labels, batch_loss = _run_iter(x,y)
        if model =='train':
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        
        loss += batch_loss.item()
        MeanRecallScore.update( o_labels.cpu() ,y )

        trange.set_postfix(loss=loss/(i+1), score=MeanRecallScore.print_score() )

    if mode=='train':
        history['train'].append({'loss':loss/len(trange)})
    
    else:
        history['valid'].append({'loss':loss /len(trange)})
    
    trange.close()


def _run_iter(x,y):
    input_images = x.to(device)
    labels = y.to(device).to(dtype=torch.float32)
    o_labels = model(input_images).to(device)
    o_labels = o_labels.to(dtype=torch.float32)
    # print('_run_iter size o_labels: ', o_labels.size(), '_run_iter size labels: ', labels.size() , '\n')
    l_loss = criteria(o_labels, labels)

    return o_labels, l_loss


def save(epoch):
    if not os.path.exists(os.path.join(CWD,'model_CNN')):
        os.makedirs(os.path.join(CWD,'model_CNN'))
    torch.save(model.state_dict(), os.path.join( CWD,'model_CNN/model.pkl.'+str(epoch) ))
    with open( os.path.join( CWD,'model_CNN/history.json'), 'w') as f:
        json.dump(history, f, indent=4)


for epoch in range(max_epoch):
    print('Epoch: {}'.format(epoch))
    _run_epoch(epoch, 'train')


'''
for batch_idx, data in enumerate(train_dataloader):
    print (data['labels'])
    # print('\n')
    # print (data['image'])


for batch_idx, data in enumerate(test_dataloader):
	print (data['image'].size())
'''

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