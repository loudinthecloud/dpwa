'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable


import sys
ext_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../")
sys.path.append(ext_path)
from dpwa.adapters.pytorch import DpwaPyTorchAdapter


import logging
def init_logging(filename):
    # Create the logs directory
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    # Init logging to file
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        filename="./logs/%s" % filename,
                        filemode='w',
                        level=logging.DEBUG)

    # logging.getLogger("dpwa.conn").setLevel(logging.INFO)


LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=128, help='Set the batch size (default: 128)')
parser.add_argument('--config-file', type=str, required=True, help='Dpwa configuration file')
parser.add_argument('--name', type=str, required=True, help="This worker's name within config file")

args = parser.parse_args()

init_logging(args.name + ".log")

batch_size = args.batch_size
print("Using batch_size =", batch_size)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


conn = DpwaPyTorchAdapter(net, args.name, args.config_file)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


MOVING_AVG_SIZE = 10

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    accuracies = []
    losses = []
    loss_mean = 9999
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx > MOVING_AVG_SIZE:
            conn.update_send(loss_mean)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate the loss
        losses += [loss.data[0]]
        loss_mean = np.array(losses[-MOVING_AVG_SIZE:]).mean()
        if batch_idx > MOVING_AVG_SIZE:
            conn.update_wait(loss_mean)

        # Calculate the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = predicted.eq(targets.data).cpu().sum()
        accuracies += [correct/total]
        accuracy = np.array(accuracies[-MOVING_AVG_SIZE:]).mean() * 100.0

        # Show progress
        progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%%" % \
                   (args.name, epoch, batch_idx, loss_mean, accuracy)
        print(progress)
        LOGGER.info(progress)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress = "[%s] E%d | B%d | Loss: %.3f | Acc: %.3f%% (%d/%d)" % \
                   (args.name, epoch, batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total)
        print(progress)
        LOGGER.info(progress)
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
