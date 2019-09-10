# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import sklearn.metrics as sm
import random
import numpy as np

from wideresnet import WideResNet, VNet
from resnet import ResNet32,VNet
from load_corrupted_data import CIFAR10, CIFAR100

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=20000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

#os.environ['CUD_DEVICE_ORDER'] = "1"
#ids = [1]


best_prec1 = 0

#use_cuda = True
#device = torch.device("cuda" if use_cuda else "cpu")


def main():
    global args, best_prec1
    args = parser.parse_args()
    print()
    print(args)

    train_loader, train_meta_loader, test_loader = build_dataset()
    # create model
    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)


    vnet = VNet(1, 100, 1).cuda()

    optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    model_loss = []
    meta_model_loss = []
    smoothing_alpha = 0.9

    meta_l = 0
    net_l = 0
    accuracy_log = []
    train_acc = []

    for iters in range(args.iters):
        adjust_learning_rate(optimizer_a, iters + 1)
        # adjust_learning_rate(optimizer_c, iters + 1)
        model.train()

        input, target = next(iter(train_loader))
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        meta_model = build_model()

        meta_model.load_state_dict(model.state_dict())
        y_f_hat = meta_model(input_var)
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))


        v_lambda = vnet(cost_v.data)

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta,(meta_model.params()),create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(iters >= 18000)) * (0.1 ** int(iters >= 19000)))  # For WRN-28-10
        #meta_lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
        meta_model.update_params(lr_inner=meta_lr,source_params=grads)
        del grads



        input_validation, target_validation = next(iter(train_meta_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation.type(torch.LongTensor), requires_grad=False)

        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]


        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()


        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        l_f = torch.sum(cost_v * w_v)


        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_model_loss.append(meta_l / (1 - smoothing_alpha ** (iters + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        model_loss.append(net_l / (1 - smoothing_alpha ** (iters + 1)))


        if (iters + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (iters + 1) // 500 + 1, args.epochs, iters + 1, args.iters, model_loss[iters],
                      meta_model_loss[iters], prec_train, prec_meta))

            losses_test = AverageMeter()
            top1_test = AverageMeter()
            model.eval()


            for i, (input_test, target_test) in enumerate(test_loader):
                input_test_var = to_var(input_test, requires_grad=False)
                target_test_var = to_var(target_test, requires_grad=False)

                # compute output
                with torch.no_grad():
                    output_test = model(input_test_var)
                loss_test = criterion(output_test, target_test_var)
                prec_test = accuracy(output_test.data, target_test_var.data, topk=(1,))[0]

                losses_test.update(loss_test.data.item(), input_test_var.size(0))
                top1_test.update(prec_test.item(), input_test_var.size(0))

            print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1_test))

            accuracy_log.append(np.array([iters, top1_test.avg])[None])
            train_acc.append(np.array([iters, prec_train])[None])

            best_prec1 = max(top1_test.avg, best_prec1)

    #np.save('meta_model_loss_%s_%s.npy' % (args.dataset, args.label_corrupt_prob), meta_model_loss)
    #np.save('model_loss_%s_%s.npy' % (args.dataset, args.label_corrupt_prob), model_loss)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    ax1, ax2, ax3 = axes.ravel()

    ax1.plot(meta_model_loss, label='meta_model_loss')
    ax1.plot(model_loss, label='model_loss')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    acc_log = np.concatenate(accuracy_log, axis=0)
    train_acc_log = np.concatenate(train_acc, axis=0)
    #np.save('L2SPL_train_acc.npy', train_acc_log)
    #np.save('L2SPL_val_acc.npy', acc_log)
    # lr_log = np.concatenate(lr_log, axis=0)

    ax2.plot(acc_log[:, 0], acc_log[:, 1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iteration')

    ax3.plot(train_acc_log[:, 0], train_acc_log[:, 1])
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Iteration')

    plt.show()



def build_dataset():
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader


def build_model():
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                       args.widen_factor, dropRate=args.droprate)
    # weights_init(model)

    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.params()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model




def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def adjust_learning_rate(optimizer, iters):

    lr = args.lr * ((0.1 ** int(iters >= 18000)) * (0.1 ** int(iters >= 19000)))  # For WRN-28-10
    #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
    # log to TensorBoard
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

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



if __name__ == '__main__':
    main()
