#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:42:46 2018

@author: rowantseng
"""

import argparse


# Parse arguments
parser = argparse.ArgumentParser(description='AMD OCT Training')

# Datasets
parser.add_argument('-d', '--data', default='path to images', type=str)
parser.add_argument('--train', default='path to train.csv', type=str)
parser.add_argument('--val', default='path to val.csv', type=str)
parser.add_argument('--test', default='path to test.csv', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize (default: 64)')
parser.add_argument('--val-batch', default=64, type=int, metavar='N',
                    help='val batchsize (default: 64)')
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn')
parser.add_argument('--cls-def', default='1234', type=str, 
                    help='def of class')

# Optimizer
parser.add_argument('--optim', default='SGD', type=str, help='def of optimizer type (default SGD)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Generate predict report
parser.add_argument('--report', default=False, type=bool, 
                    help='set \"True\" if you want to generate inference result report')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
