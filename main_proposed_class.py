# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import torch.nn.utils.prune as prune
# Custom Libraries
import utils
import random
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--method", default="proposed_prune_ratio_fixed", help="the proposed method")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=60, type=int)
parser.add_argument("--start_prune_round", default=0, type=int)
parser.add_argument("--train_epochs", default=200, type=int)
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--early_stop", default=80, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--prune_type", default="local", type=str, help="local | global")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | advanced_dropout_fc| lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--initial_percent", default=100, type=float, help='percentage of the weights that is trainable and initialized')
parser.add_argument("--prune_rounds", default=1, type=int, help="Pruning args.prune_roundss count")
parser.add_argument("--prune_ratio", default=1, type=float, help="Prune ratio during train")
parser.add_argument("--prune_prob", default=0.01, type=float, help="probability to prune during train")
parser.add_argument("--target_ratio", default=2, type=float, )
parser.add_argument("--optimizer", default="sgd", help="adam | sgd")
parser.add_argument("--weight_decay", default=1.2e-3, type=float, help="weight decay for adam optim")
parser.add_argument("--seed", default=1, type=int)

args = parser.parse_args()
torch.cuda.manual_seed_all(args.seed)
class Proposed_prune():
    def __init__(self, args) -> None:
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args


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
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(self.mean[args.dataset], self.std[args.dataset])])
        if args.dataset == "mnist":
            traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
            testdataset = datasets.MNIST('../data', train=False, transform=transform)
            split = [55000, 5000]
            from archs.mnist import AlexNet, LeNet5, fc1, advanced_dropout_fc, vgg, resnet

        elif args.dataset == "cifar10":
            traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
            testdataset = datasets.CIFAR10('../data', train=False, transform=transform)   
            split = [45000, 5000]   
            from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

        elif args.dataset == "fashionmnist":
            traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
            testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
            from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

        elif args.dataset == "cifar100":
            traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
            testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
            from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
        else:
            print('Wrong dataset choice')
            exit()
        
        train_dataset, val_dataset = torch.utils.data.random_split(traindataset, split, generator=torch.Generator().manual_seed(args.seed))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
        #train_loader = cycle(train_loader)
        self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=True)
        if args.arch_type == "fc1":
            self.model = fc1.fc1().to(self.device)
        elif args.arch_type == "advanced_dropout_fc":
            self.model = advanced_dropout_fc.advanced_drop_fc().to(self.device)
        elif args.arch_type == "lenet5":
            self.model = LeNet5.LeNet5().to(self.device)
        elif args.arch_type == "alexnet":
            self.model = AlexNet.AlexNet().to(self.device)
        elif args.arch_type == "vgg16":
            self.model = vgg.vgg16().to(self.device)  
        elif args.arch_type == "resnet18":
            self.model = resnet.resnet18().to(self.device)   
        elif args.arch_type == "densenet121":
            self.model = densenet.densenet121().to(self.device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()
        self.parameters_to_prune = []
        self.initial_num_weights = []
        self.target_num_weight = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.parameters_to_prune.append((module, 'weight'))
        self.parameters_to_prune = tuple(self.parameters_to_prune)

        # self.total_num_weights = sum(p.numel() for name, p in self.model.named_parameters() if 'weight' in name)
        # self.target_num_weights = args.target_ratio * self.total_num_weights
        if args.prune_type == 'local':
            # i = 0
            for layer, name in self.parameters_to_prune:
                # i += 1
                # if i != len(self.parameters_to_prune):
                prune.random_unstructured(layer, name=name, amount=1.0-args.initial_percent*0.01)
        elif args.prune_type == 'global':
            prune.global_unstructured(self.parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=1.0-args.initial_percent*0.01)
        else:
            raise Exception('Invalid prune type.')
        self.save_path = f"{os.getcwd()}/saves/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_percent_{args.initial_percent}/{args.dataset}/"
        self.plot_path = f"{os.getcwd()}/plots/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_percent_{args.initial_percent}/{args.dataset}/"
        utils.checkdir(self.save_path)
        utils.checkdir(self.plot_path)
        with open(os.path.join(self.save_path, "args.txt"), 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f) 
    
    def prune(self, ):
        # plot the histogram of the initial distribution
        for name, p in self.model.named_parameters():
            weight_nz = p[torch.nonzero(p, as_tuple=True)]
            plt.hist(weight_nz.cpu().data.view(-1), bins=30)
            plt.savefig(os.path.join(self.plot_path, f"{name}_initial.png"))
            plt.close()
        writer = SummaryWriter(self.save_path)
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            raise Exception('wrong optimizer, has to be adam or sgd')
        for name, param in self.model.named_parameters():
            print(name, param.size())
        bestacc = 0.0
        best_accuracy = 0
        comp = np.zeros(args.train_epochs,float)
        bestacc = np.zeros(args.train_epochs,float)
        testacc = np.zeros(args.train_epochs, float)
        sparsity_ = np.zeros(args.train_epochs, float)
        step = 0
        all_loss = np.zeros(args.train_epochs,float)
        all_accuracy = np.zeros(args.train_epochs,float)
        early_stop_trigger = 0

            # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(self.model, writer, 0)
        sparsity = round(100.0-comp1, 1)
        sparsity_[0] = sparsity
        comp[0] = comp1
        pbar = tqdm(range(args.train_epochs))

        for train_epoch in pbar:
            # Frequency for Testing
            if train_epoch % args.valid_freq == 0:
                val_accuracy = self.test(self.model, self.val_loader, criterion)

                # Save Weights
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    early_stop_trigger = 0
                else:
                    early_stop_trigger += 1
            
            # Training
            loss = self.train(self.model, self.train_loader, optimizer, criterion, self.args)
            torch.save(self.model, os.path.join(self.save_path, f"{train_epoch}_model_{args.prune_type}.pt"))
            all_loss[train_epoch] = loss
            all_accuracy[train_epoch] = val_accuracy
            # Frequency for Printing Accuracy and Loss
            if train_epoch % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} Loss: {loss:.6f} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            if early_stop_trigger > args.early_stop:
                break

            comp1 = utils.print_nonzeros(self.model, writer, train_epoch)
            sparsity_[train_epoch] = round(100.0-comp1, 1)
            # best_val_model = torch.load(os.path.join(self.save_path, f"{train_epoch}_model_{args.prune_type}.pt"))
            test_accuracy = self.test(self.model, self.test_loader, criterion)
            print(f'Test Accuracy: {test_accuracy}')
            writer.add_scalar('Accuracy_sparsity/val', best_accuracy, round(100.0-comp1, 1))
            writer.add_scalar('Accuracy_sparsity/test', test_accuracy, round(100.0-comp1, 1))
            writer.add_scalar('Accuracy_epoch/test', test_accuracy, train_epoch)
            bestacc[0] = best_accuracy
            testacc[train_epoch] = test_accuracy
            fig = utils.plot_sparsity_testacc(sparsity_[:train_epoch+1], testacc[:train_epoch+1], self.plot_path)
            writer.add_figure('sparsity_testacc', fig, train_epoch)
            if comp1 < 2.: 
                break
        for name, p in self.model.named_parameters():
            weight_nz = p[torch.nonzero(p, as_tuple=True)]
            plt.hist(weight_nz.cpu().data.view(-1), bins=30)
            plt.savefig(os.path.join(self.plot_path, f"{name}.png"))
            plt.close()



    def train(self, model, train_loader, optimizer, criterion, args):
        metric = MeanMetric()
        EPS = 1e-6
        model.train()
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            metric.update(train_loss.detach().cpu())
            # Freezing Pruned weights by making their gradients Zero
            for name, p in model.named_parameters():
                if 'weight' in name:
                    tensor = p.data
                    grad_tensor = p.grad.data
                    grad_tensor = torch.where(torch.abs(tensor) < EPS, 0, grad_tensor)
                    p.grad.data = grad_tensor.to(self.device)

            optimizer.step()
            if batch_idx % 100 == 0:
                self.prune_and_reconnect(model, args.prune_prob, args.prune_ratio * 0.01)
        
        return metric.compute().item()

    def test(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy
    
    def prune_and_reconnect(self, model, prune_prob, prune_ratio):
        prune_count = []
        mask_bef_pru = []
        for name, param in model.named_buffers():
            prune_count.append(int(np.rint((torch.count_nonzero(param.data)).cpu().numpy() * prune_ratio)))
            mask_bef_pru.append(param.data)
        # prune the connections
        if args.prune_type == 'local':
            i = 0
            for layer, name in self.parameters_to_prune:
                i +=1
                if i == len(self.parameters_to_prune):
                    prune.l1_unstructured(layer, name=name, amount=prune_ratio/2)
                else:
                    # prune at half ratio for the last output layer
                    prune.l1_unstructured(layer, name=name, amount=prune_ratio)
        elif args.prune_type == 'global':
            prune.global_unstructured(self.parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.prune_percent)
        # if prune_count[2] / 2 > 1:
        # else:
            # prune.l1_unstructured(model.fc3, name='weight', amount=1)
        # reconnect the zeroed connections
        i = 0
        for module in list(model.children()):
            mask = mask_bef_pru[i]
            if i == len(self.parameters_to_prune) - 1:
                num_reconnect = prune_count[i] // 4 
            else:
                num_reconnect = prune_count[i] // 2 # add less connection than prune
            zero_indices = (mask.view(-1) == 0).nonzero()
        #     # shuffle it
            idx = torch.randperm(zero_indices.nelement())
            zero_indices = zero_indices.view(-1)[idx].view(zero_indices.size())[0:num_reconnect]
    
            (list(module.named_buffers())[0][1]).view(-1)[zero_indices] = 1.
            i += 1

def main():
    proposed_prune = Proposed_prune(args)
    proposed_prune.prune()
if __name__ == "__main__":
    main()

        

