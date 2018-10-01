from __future__ import division
from __future__ import print_function


import torch
import os
import argparse

from torch import optim
from datasets import get_mnist_dataset, get_cifar10_dataset, get_data_loader
from utils import *
from parser_utils import str2bool
from models import resnet, pyt_resnet
from adversary import Solver

def main(args):

    # setting device
    if args.cuda and torch.cuda.is_available():
        """
        if argument is given and cuda is available
        """
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_name = args.dataset.lower()
    if 'mnist' not in dataset_name and 'cifar' not in dataset_name:
        raise Exception('{} dataset not yet supported'.format(dataset_name))
    else:
        dset = get_cifar10_dataset if 'cifar' in dataset_name else get_mnist_dataset
    # getting the right dataset
    train,test = dset()
    dataloader = get_data_loader(train,test,batch_size = args.batch_size)
    trainloader, testloader = dataloader['train'],dataloader['test']
    batch, labels = next(iter(trainloader))
    # converting data to tensors
    batch_var = Variable(batch).to(device)
    labels_var = Variable(one_hotify(labels).to(device))
    
    # getting the right model
    model_name = args.model.lower()
    model_dict = {'resnet18':resnet.resnet18(num_classes = 10, version = 2),
                            'resnet34':resnet.resnet34(num_classes = 10, version = 2),
                            'resnet50':resnet.resnet50(num_classes = 10, version = 2),
                            'resnet101':resnet.resnet101(num_classes = 10, version = 2),
                            'resenet152':resnet.resnet152(num_classes = 10, version = 2),
                            'capsnet':None}
    if model_name not in model_dict:
        raise Exception('{} not implemented, try main.py --help for additional information'.foramt(model_name))
    else:
        model = model_dict[model_name]
    # temporary
    if not model:
        raise Exception('CapsNet in progress')
    ckpt_name = os.path.join(args.ckpt_dir,
                                '{}_{}'.format(model_name,dataset_name)+'.pth.tar')
    if os.path.isfile(ckpt_name):
        base_trainer.load(filename = ckpt_name)    
    model.to(device)#model graph is placed
    fname = os.path.join('checkpoints','{}_{}'.format('resnet18_v2','cifar10')+'.pth.tar')
    base_loss = nn.CrossEntropyLoss()
    base_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    base_trainer = Trainer(model, base_optimizer, base_loss,
                        trainloader, testloader, use_cuda=args.cuda)
    base_trainer.load_checkpoint(fname)
    # base_trainer.run(epochs=1)
    #base_trainer.save_checkpoint(ckpt_name)
    net = Solver(args,model,dataloader)
    net.generate(num_sample=args.batch_size,
                    target=args.target,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    iteration=args.iteration)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack and model arguments')
    parser.add_argument('--model',type=str,default='resnet18',help='resnet18,resnet34,capsnet,etc..')
    parser.add_argument('--epoch', type=int, default=15, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--y_dim', type=int, default=10, help='the number of classes')
    parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    parser.add_argument('--eps', type=float, default=1e-9, help='epsilon')
    parser.add_argument('--env_name', type=str, default='main', help='experiment name')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset type')
    parser.add_argument('--dset_dir', type=str, default='datasets', help='dataset directory path')
    parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='', help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--mode', type=str, default='train', help='train / test / generate / universal')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    parser.add_argument('--tensorboard', type=str2bool, default=False, help='enable tensorboard')
    parser.add_argument('--visdom', type=str2bool, default=False, help='enable visdom')
    parser.add_argument('--visdom_port', type=str, default=55558, help='visdom port')
    args = parser.parse_args()

    main(args)    
