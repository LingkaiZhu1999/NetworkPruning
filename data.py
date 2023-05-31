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
        'cifar100': (0.5071, 0.4867, 0.4408),
        'imagenet': (0.485, 0.456, 0.406)
        }
        self.std = {
        'mnist': (0.3081,),
        'cifar10': (0.2470, 0.2435, 0.2616),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'imagenet': (0.229, 0.224, 0.225)
        }
        self.seed = seed

    def get_dataset(self, dataset, val=True):
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(self.mean[dataset], self.std[dataset])])
        if exists(f'../data/{dataset}_train.pt') and val == True:
            return torch.load(f'../data/{dataset}_train.pt'), torch.load(f'../data/{dataset}_val.pt'), torch.load(f'../data/{dataset}_test.pt')
        else:
            if dataset == "mnist":
                traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
                testdataset = datasets.MNIST('../data', train=False, transform=transform)
                split = [55000, 5000]
                train_dataset, val_dataset = torch.utils.data.random_split(traindataset, split, generator=torch.Generator().manual_seed(self.seed))
                torch.save(train_dataset, f'../data/{dataset}_train.pt'), torch.save(val_dataset, f'../data/{dataset}_val.pt'), torch.save(testdataset, f'../data/{dataset}_test.pt')
                return train_dataset, val_dataset, testdataset
            elif dataset == "cifar10":
                transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[dataset], self.std[dataset]),
                ])
                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean[dataset], self.std[dataset]),
                ])
                traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform_train)
                testdataset = datasets.CIFAR10('../data', train=False, transform=transform_test)  
                if val==True: 
                    split = [45000, 5000] 
                    train_dataset, val_dataset = torch.utils.data.random_split(traindataset, split, generator=torch.Generator().manual_seed(self.seed))
                    torch.save(train_dataset, f'../data/{dataset}_train.pt'), torch.save(val_dataset, f'../data/{dataset}_val.pt'), torch.save(testdataset, f'../data/{dataset}_test.pt')  
                    return train_dataset, val_dataset, testdataset
                else:
                    return traindataset, testdataset
            elif dataset == "fashionmnist":
                traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
                testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)

            elif dataset == "cifar100":
                traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
                testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
                if val==True: 
                    split = [45000, 5000] 
                    train_dataset, val_dataset = torch.utils.data.random_split(traindataset, split, generator=torch.Generator().manual_seed(self.seed))
                    torch.save(train_dataset, f'../data/{dataset}_train.pt'), torch.save(val_dataset, f'../data/{dataset}_val.pt'), torch.save(testdataset, f'../data/{dataset}_test.pt')  
                    return train_dataset, val_dataset, testdataset
                else:
                    return traindataset, testdataset  

            elif dataset == "imagenet":
                transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[dataset], self.std[dataset]),
                ])

                transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[dataset], self.std[dataset]),
                ]) 
                train_dataset = datasets.ImageNet('../data', split='train', download=True,transform=transform_train)
                val_dataset = datasets.ImageNet('../data', split='val', transform=transform_val) 
                # devkit = datasets.ImageNet('../data', split='devkit')
                return train_dataset, val_dataset 
            else:
                print('Wrong dataset choice')
                exit()

            
