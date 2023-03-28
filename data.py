import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from os.path import exists


class Data():
    def __init__(self, seed) -> None:
        self.mean = {
        'mnist': (0.1307,),
        'cifar10': (0.4914, 0.4822 ,0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408)
        }
        self.std = {
        'mnist': (0.3081,),
        'cifar10': (0.2470, 0.2435, 0.2616),
        'cifar100': (0.2675, 0.2565, 0.2761),
        }
        self.seed = seed

    def get_dataset(self, dataset):
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(self.mean[dataset], self.std[dataset])])
        if exists(f'../data/{dataset}_train.pt'):
            print('True')
            return torch.load(f'../data/{dataset}_train.pt'), torch.load(f'../data/{dataset}_val.pt'), torch.load(f'../data/{dataset}_test.pt')
        else:
            if dataset == "mnist":
                traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
                testdataset = datasets.MNIST('../data', train=False, transform=transform)
                split = [55000, 5000]
                from archs.mnist import AlexNet, LeNet5, fc1, advanced_dropout_fc, vgg, resnet

            elif dataset == "cifar10":
                traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
                testdataset = datasets.CIFAR10('../data', train=False, transform=transform)   
                split = [45000, 5000]   
                from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

            elif dataset == "fashionmnist":
                traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
                testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
                from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

            elif dataset == "cifar100":
                traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
                testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
                from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
            else:
                print('Wrong dataset choice')
                exit()
        
            train_dataset, val_dataset = torch.utils.data.random_split(traindataset, split, generator=torch.Generator().manual_seed(self.seed))
            torch.save(train_dataset, f'../data/{dataset}_train.pt'), torch.save(train_dataset, f'../data/{dataset}_val.pt'), torch.save(train_dataset, f'../data/{dataset}_test.pt')

            return train_dataset, val_dataset, testdataset
