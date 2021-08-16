import torch
import torch.nn as nn
import math
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import argparse
import pdb
import random
import numpy as np
import os

from models.googlenet import GoogLeNet
from models.googlenet_batchnorm import GoogLeNet as GoogLeNet_BN
from models.resnet import ResNet18, ResNet34
from models.vgg import VGG13
from tensorboardX import SummaryWriter


'''
Fuctions
'''

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for CIFAR10 traing")

    parser.add_argument("--model", type=str, default=None,
                        help="model name")
    parser.add_argument("--eval-period", type=int, default=10,
                        help="eval period")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="eval period")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="exp name")
    return parser.parse_args()

def train_epoch(model, train_loader):
    model.train()
    print("{}th epoch starting.".format(epoch))
    for i, (images, labels) in enumerate(train_loader) :
        images, labels = images.to(device), labels.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        train_loss = loss_function(model(images), labels)
        train_loss.backward()

        optimizer.step()

    print ("Epoch [{}] Loss: {:.4f}".format(epoch+1, train_loss.item()))
    writer.add_scalar('sch_loss/train_loss',train_loss.item(),epoch+1)

def eval_epoch(model, test_dataset):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader :
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            test_loss += loss_function(output, labels).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            total += labels.size(0)

    print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss /total, correct, total,
            100. * correct / total))
    writer.add_scalar('sch_acc/test_acc',100. * correct / total,epoch+1)
    writer.add_scalar('sch_loss/test_loss',test_loss/total,epoch+1)
    


'''

'''


args = get_arguments()
print('Called with args:')
print(args)
# pdb.set_trace()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

exp_log_dir = os.path.join('./logs', args.exp_name)
writer=SummaryWriter(log_dir=exp_log_dir)
'''
Step 1:
'''

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

train_dataset = datasets.CIFAR10(root='./cifar_10data/',
                                 train=True, 
                                 transform=transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='./cifar_10data/',
                                train=False, 
                                transform=test_transform)
    
'''
Step 2
여기에서 lr, weight decay, batch_size, epoch을 조절해 주었음
#DEFAULT를 통해 원래 값을 입력해주어 헷갈리지 않도록 했음
'''
if args.model == 'googlenet':
    model = GoogLeNet().to(device)
    train_args = {'lr':0.1, 'weight_decay':5e-4, 'batch_size':100, 'max_epoch': 100}
    #DEFAULT train_args = {'lr':0.1, 'weight_decay':5e-4, 'batch_size':100, 'max_iter': 100}
elif args.model == 'googlenet_batchnorm':
    model = GoogLeNet_BN().to(device)
    train_args = {'lr':0.1, 'weight_decay':2e-4, 'batch_size':50, 'max_epoch': 150}
    #DEFAULT train_args = {'lr':0.1, 'weight_decay':5e-4, 'batch_size':100, 'max_iter': 100}
elif args.model == 'resnet18':
    model = ResNet18().to(device)
    train_args = {'lr':0.03, 'weight_decay':5e-4, 'batch_size':100, 'max_epoch': 150}
    #DEFAULT train_args = {'lr':0.03, 'weight_decay':5e-4, 'batch_size':100, 'max_iter': 100}
elif args.model == 'resnet34':
    model = ResNet34().to(device)
    train_args = {'lr':0.03, 'weight_decay':5e-4, 'batch_size':25, 'max_epoch': 150}
    #DEFAULT train_args = {'lr':0.03, 'weight_decay':5e-4, 'batch_size':100, 'max_iter': 100}
elif args.model == 'vgg':
    model = VGG13().to(device)
    train_args = {'lr':0.03, 'momentum':0.9, 'weight_decay':5e-4, 'batch_size':256, 'max_epoch': 200}
    #DEFAULT train_args = {'lr':0.05, 'momentum':0.9, 'weight_decay':5e-4, 'batch_size':128, 'max_iter': 200}
print(train_args)
'''
Step 3
'''
# model = VGG13().to(device)
loss_function = torch.nn.CrossEntropyLoss()

if args.model == 'vgg':
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args['lr'], momentum=train_args['momentum'], weight_decay=train_args['weight_decay'])
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
#scheduler의 추가로 정확도가 눈에 띄게 올랐음
'''
Step 4
'''
model.train()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_args['batch_size'], shuffle=True)

import time
start = time.time()
for epoch in range(train_args['max_epoch']) :
    # pdb.set_trace()
    train_epoch(model, train_loader)
    if epoch % args.eval_period == 0:
        eval_epoch(model, test_dataset)
    scheduler.step()

eval_epoch(model, test_dataset)

end = time.time()
print("Time ellapsed in training is: {}".format(end - start))