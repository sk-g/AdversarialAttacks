import torch

from torch import optim
from datasets import get_mnist_dataset, get_cifar10_dataset, get_data_loader
from utils import *

from models import resnet, pyt_resnet


trainset, testset = get_cifar10_dataset()
trainloader, testloader = get_data_loader(trainset, testset)
batch, labels = next(iter(trainloader))
plot_batch(batch)
batch_var = Variable(batch.cuda())

base_model = resnet.resnet18(num_classes=10, version = 2).cuda()
print(count_params(base_model))

base_loss = nn.CrossEntropyLoss()
base_optimizer = optim.SGD(base_model.parameters(), lr=0.1)
base_trainer = Trainer(base_model, base_optimizer, base_loss,
                       trainloader, testloader, use_cuda=True)
labels_var = Variable(one_hotify(labels).cuda())

# RandomCrop(28) reduces accuracy considerably with this architecture
base_trainer.run(epochs=10)
for param_group in base_trainer.optimizer.param_groups:
    param_group['lr'] = 0.1 * param_group['lr']
base_trainer.run(epochs=5)
for param_group in base_trainer.optimizer.param_groups:
    param_group['lr'] = 0.1 * param_group['lr']
base_trainer.run(epochs=5)
base_trainer.save_checkpoint('weights/resnet_cifar.pth.tar')


base_model = resnet.resnet18(num_classes=10, version = 1)#.cuda()
print(count_params(base_model))

base_loss = nn.CrossEntropyLoss()
base_optimizer = optim.SGD(base_model.parameters(), lr=0.1)
base_trainer = Trainer(base_model, base_optimizer, base_loss,
                       trainloader, testloader, use_cuda=False)
labels_var = Variable(one_hotify(labels))#.cuda())

# RandomCrop(28) reduces accuracy considerably with this architecture
base_trainer.run(epochs=10)
for param_group in base_trainer.optimizer.param_groups:
    param_group['lr'] = 0.1 * param_group['lr']
base_trainer.run(epochs=5)
for param_group in base_trainer.optimizer.param_groups:
    param_group['lr'] = 0.1 * param_group['lr']
base_trainer.run(epochs=5)
base_trainer.save_checkpoint('weights/resnet_cifar.pth.tar')