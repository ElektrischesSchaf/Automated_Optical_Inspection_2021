import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from PIL import Image
import numpy as np
import pandas as pd

from tqdm import tqdm_notebook as tqdm
from tqdm import trange

import os

CWD = os.getcwd()
# if 'task1' not in CWD:
#     CWD = os.path.join(CWD, 'task1')


learning_rate = 1e-5
max_epoch = 3
batch_size = 64
batch_size_test = 64
threshold = 0.5

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
            out_labels=torch.tensor([1,0,0,0,0,0,0])
        if labels==1:
            out_labels=torch.tensor([0,1,0,0,0,0,0])
        if labels==2:
            out_labels=torch.tensor([0,0,1,0,0,0,0])
        if labels==3:
            out_labels=torch.tensor([0,0,0,1,0,0,0])
        if labels==4:
            out_labels=torch.tensor([0,0,0,0,1,0,0])
        if labels==5:
            out_labels=torch.tensor([0,0,0,0,0,1,0])
        if labels==6:
            out_labels=torch.tensor([0,0,0,0,0,0,1])

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

        return image

    def __len__(self):
        return len(self.name_frame)


train_dataset = MyTrainDataset( text_file= './dataset/train.csv' , root_dir='./dataset/train_images/train_images/')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = MyTestDataset( text_file= './dataset/test.csv' , root_dir='./dataset/test_images/test_images/')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

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

        self.linear = nn.Linear( 1*254*254, 7 )

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

        out= torch.softmax(out, dim=1)

        _, max_index = torch.max(out, dim=1)
        # print(max_index)

        # https://stackoverflow.com/questions/55549843/pytorch-doesnt-support-one-hot-vector
        # TODO move this out after loss func
        y = torch.zeros(out.size(0), out.size(1))
        y[range(y.shape[0]), max_index]=1
        # print(y)
        
        return y


#  batch_size,  in_channels, out_channels, kernel_heights, stride, padding, 
model = CNN_Simple(batch_size, 3, 1, [3], 1, 0)


opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criteria = torch.nn.BCELoss() # BCELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
history = {'train':[],'valid':[]}

# sigma((number of correct predictions of class)/(number of total images of class)) / (Number of classes)
# TODO fix 2021-05-30
class mean_recall(): # TODO fix 
    def __init__(self):
        self.n_corrects_class_0 = 0
        self.n_corrects_class_1 = 0
        self.n_corrects_class_2 = 0
        self.n_corrects_class_3 = 0
        self.n_corrects_class_4 = 0
        self.n_corrects_class_5 = 0
        self.n_corrects_class_6 = 0

        self.n_recall_class_0 = 0
        self.n_recall_class_1 = 0
        self.n_recall_class_2 = 0
        self.n_recall_class_3 = 0
        self.n_recall_class_4 = 0
        self.n_recall_class_5 = 0
        self.n_recall_class_6 = 0

        # self.n_correct_predictions=0
        # self.n_recall=0
        # self.n_corrects=0
    
        # the prediction contain this class or not, 0 or 1
        self.n_classes_0 = 0
        self.n_classes_1 = 0
        self.n_classes_2 = 0
        self.n_classes_3 = 0
        self.n_classes_4 = 0
        self.n_classes_5 = 0
        self.n_classes_6 = 0

    def reset(self):
        self.n_predictions_class_0 = 0
        self.n_predictions_class_1 = 0
        self.n_predictions_class_2 = 0
        self.n_predictions_class_3 = 0
        self.n_predictions_class_4 = 0
        self.n_predictions_class_5 = 0
        self.n_predictions_class_6 = 0

        self.n_classes_0 = 0
        self.n_classes_1 = 0
        self.n_classes_2 = 0
        self.n_classes_3 = 0
        self.n_classes_4 = 0
        self.n_classes_5 = 0
        self.n_classes_6 = 0

    def update(self, predicts, ground_truth):
        
        # calculate number of total images of each class and number of classes
        for i in range(ground_truth.size(0)):
            # print('test ', ground_truth[i].numpy() , '\n')
            if np.array_equal(ground_truth[i].numpy(),[1,0,0,0,0,0,0]):
                self.n_recall_class_0 += 1
                self.n_classes_0 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,1,0,0,0,0,0]):
                self.n_recall_class_1 += 1
                self.n_classes_1 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,0,1,0,0,0,0]):
                self.n_recall_class_2 += 1
                self.n_classes_2 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,0,0,1,0,0,0]):
                self.n_recall_class_3 += 1
                self.n_classes_3 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,0,0,0,1,0,0]):
                self.n_recall_class_4 += 1
                self.n_classes_4 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,0,0,0,0,1,0]):
                self.n_recall_class_5 += 1
                self.n_classes_5 = 1
            if np.array_equal(ground_truth[i].numpy(),[0,0,0,0,0,0,1]):
                self.n_recall_class_6 += 1
                self.n_classes_6 = 1
        
        for i in range(predicts.size(0)):
            if np.array_equal(ground_truth[i].numpy, [1,0,0,0,0,0,0]):
                self.n_corrects_class_0 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,1,0,0,0,0,0]):
                self.n_corrects_class_1 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,0,1,0,0,0,0]):
                self.n_corrects_class_2 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,0,0,1,0,0,0]):
                self.n_corrects_class_3 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,0,0,0,1,0,0]):
                self.n_corrects_class_4 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,0,0,0,0,1,0]):
                self.n_corrects_class_5 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()
            if np.array_equal(ground_truth[i].numpy, [0,0,0,0,0,0,1]):
                self.n_corrects_class_6 += torch.sum(groundTruth[i].type(torch.bool) * predicts[i]).data.item()

    def get_score(self):
        recall_class_0=0
        if self.n_recall_class_0!=0:
            recall_class_0 = self.n_corrects_class_0 / self.n_recall_class_0
        recall_class_1=0
        if self.n_recall_class_1!=0:
            recall_class_1 = self.n_corrects_class_1 / self.n_recall_class_1
        recall_class_2=0
        if self.n_recall_class_2!=0:
            recall_class_2 = self.n_corrects_class_2 / self.n_recall_class_2
        recall_class_3=0
        if self.n_recall_class_3!=0:
            recall_class_3 = self.n_corrects_class_3 / self.n_recall_class_3
        recall_class_4=0
        if self.n_recall_class_4!=0:
            recall_class_4 = self.n_corrects_class_4 / self.n_recall_class_4
        recall_class_5=0
        if self.n_recall_class_5!=0:
            recall_class_5 = self.n_corrects_class_5 / self.n_recall_class_5
        recall_class_6=0
        if self.n_recall_class_6!=0:
            recall_class_6 = self.n_corrects_class_6 / self.n_recall_class_6

        return (recall_class_0 + recall_class_1 + recall_class_2 + recall_class_3 + recall_class_4 + recall_class_5 + recall_class_6) / (self.n_classes_0+self.n_classes_1+self.n_classes_2+self.n_classes_3+self.n_classes_4+self.n_classes_5+self.n_classes_6)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)


def _run_epoch(epoch, mode):
    model.train(True)
    if mode=='train':
        description = 'train'
    if mode=='valid':
        description = 'valid'
        # TODO dataloader for valid dataset
    trange = tqdm(enumerate( train_dataloader ), total=len(train_dataloader), desc=description )

    loss = 0
    MeanRecallScore = mean_recall()

    for i, (x, y) in trange:
        # print('_run_epoch type(x) ', type(x), ' x= ', x, '\n')
        # print('_run_epoch type(y) ', type(y), ' y= ', y, '\n')
        o_labels, batch_loss = _run_iter(x, y)
        if model =='train':
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        
        loss += batch_loss.item()
        MeanRecallScore.update( o_labels.cpu() ,y )

        trange.set_postfix(loss=loss/(i+1), score = MeanRecallScore.print_score() )

    if mode=='train':
        history['train'].append({'mean_recall':MeanRecallScore.print_score(), 'loss':loss/len(trange)})
    
    else:
        history['valid'].append({'mean_recall':MeanRecallScore.print_score(), 'loss':loss/len(trange)})
        print('the mean_score: ', MeanRecallScore.print_score(), '\n')

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
    _run_epoch(epoch, 'valid')
    save(epoch)


train_loss = [l['loss'] for l in history['train']]
# valid_loss = [l['loss'] for l in history['valid']]
train_mean_recall = [l['mean_recall'] for l in history['train']]
# valid_mean_recall = [l['mean_recall'] for l in history['valid']]




best_model = max_epoch -1 # TODO fix
model.load_state_dict(state_dict=torch.load(os.path.join(CWD,'model_CNN/model.pkl.{}'.format(best_model))))
model.train(False)
model.to(device)
model.eval()

# double ckeck the best_model_score
# _run_epoch(1, 'valid')


trange = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Predict')
prediction = torch.empty((0, 7), dtype=bool)

with torch.no_grad():
    for i, x in trange:
        o_labels = model(x.to(device))
        # print('In testing: ', o_labels.size(), ' ')
        for idx, o_label in enumerate(o_labels):
            o_label = o_label.to('cpu')
            
            # TODO test remove threshold and inspect the result lables
            
            o_label = o_label > 0
            o_label = o_label.unsqueeze(0)
            print(o_label)

            # print('In testing: ' ,o_label.size(), '\n')
            # print('In testing: ' ,prediction.size(), '\n')
            
            prediction = torch.cat((prediction, o_label), 0)


print('yee: ', prediction.size() )

# prediction = torch.cat(prediction).detach().numpy().astype(int)
prediction = np.array(prediction)
print('Prediction shape: ', prediction.shape)

prediction = prediction.astype(int)

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