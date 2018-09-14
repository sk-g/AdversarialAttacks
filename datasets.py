import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms as T
from torchvision import datasets as dset


MNIST_TRN_TRANSFORM = T.Compose([
    T.ToTensor()
])
MNIST_TST_TRANSFORM = T.Compose([
    T.ToTensor()
])
CIFAR10_TRN_TRANSFORM = T.Compose([
    T.RandomCrop(28),
    T.ToTensor()
])
CIFAR10_TST_TRANSFORM = T.Compose([
    T.CenterCrop(28),
    T.ToTensor()                 
])



CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')

def get_mnist_dataset(trn_size=60000, tst_size=10000):
    trainset = dset.MNIST(root='./datasets', train=True,
                          download=True, transform=MNIST_TRN_TRANSFORM)
    trainset.train_data = trainset.train_data[:trn_size]
    trainset.train_labels = trainset.train_labels[:trn_size]
    testset = dset.MNIST(root='./datasets', train=False,
                         download=True, transform=MNIST_TST_TRANSFORM)
    testset.test_data = testset.test_data[:tst_size]
    testset.test_labels = testset.test_labels[:tst_size]
    return trainset, testset

def get_cifar10_dataset(trn_size=60000, tst_size=10000):
    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor()
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    trainset = dset.CIFAR10(root='./datasets', train=True,
                          download=True, transform=data_transforms['train'])
    trainset.train_data = trainset.train_data[:trn_size]
    trainset.train_labels = trainset.train_labels[:trn_size]
    testset = dset.CIFAR10(root='./datasets', train=False,
                         download=True, transform=data_transforms['val'])
    testset.test_data = testset.test_data[:tst_size]
    testset.test_labels = testset.test_labels[:tst_size]
    return trainset, testset

def get_data_loader(trainset, testset, batch_size=32):
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False)
    return trainloader, testloader
