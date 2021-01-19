#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:53:47 2018

@author: rowantseng
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import itertools

import operator


def preprocess_image(img, use_cuda=False):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, volatile = True)
    if use_cuda:
        input = input.cuda()
    return input


def imshow(loader, mean, std):
    """Show and unnormalize data"""
    # Make some examples
    dataiter = iter(loader)
    images, labels = dataiter.next()
    
    img = torchvision.utils.make_grid(images)
    
    std = torch.FloatTensor(np.array(std)).unsqueeze(1).unsqueeze(2)
    mean = torch.FloatTensor(np.array(mean)).unsqueeze(1).unsqueeze(2)
    mean.expand_as(img)
    std.expand_as(img)

    img = torch.addcmul(mean, 1, img, std)    # unnormalize

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
    
def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in train_loader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print('==> Mean: ')
    print (mean)
    print('==> STD: ')
    print (std)
    return mean, std


def adjust_learning_rate(state, optimizer, epoch):
    if epoch in state['schedule']:
        state['lr'] *= state['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    return optimizer


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)    
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Groundtruth')
    plt.xlabel('Prediction')


def label2cls(labels, offset):
    labels = list(map(int, labels))
    cls = list(map(operator.add, labels, offset))
    return cls


def cls2label(cls):
    cls = list(map(int, cls))
    labels = list(range(0, len(cls)))
    offset = list(map(operator.sub, cls, labels))
    return (labels, offset)
