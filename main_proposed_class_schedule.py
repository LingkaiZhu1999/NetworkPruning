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
from data import Data
from archs.cifar10 import VGG16, AlexNet, LeNet5, fc1, resnet, densenet, vgg
from prune_and_reconnect import Prune_and_Reconnect
import global_unstructure
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from ptflops import pytorch_ops
import copy
# from torchstat import stat
sns.set_style('darkgrid')

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="proposed_prune_ratio_fixed_experimental_exp_7_fine_tune", help="the proposed method")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=60, type=int)
parser.add_argument("--start_prune_round", default=0, type=int)
parser.add_argument("--train_epochs", default=160, type=int)
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--early_stop", default=None, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--prune_type", default="global", type=str, help="local | global")
parser.add_argument("--device", default="cuda:1", type=str)
parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="vgg16", type=str, help="fc1 | advanced_dropout_fc| lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--initial_percent", default=100, type=float, help='percentage of the weights that is trainable and initialized')
parser.add_argument("--prune_ratio", default=1, type=float, help="Prune ratio during train")
parser.add_argument("--prune_prob", default=0.01, type=float, help="probability to prune during train")
parser.add_argument("--target_ratio", default=1.5, type=float, )
parser.add_argument("--prune_conv1", default=False)
parser.add_argument("--output_target_ratio", default=5, type=float)
parser.add_argument("--optimizer", default="sgd", help="adam | sgd")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight decay for adam optim")
parser.add_argument("--seed", default=1, type=int)

args = parser.parse_args()
torch.cuda.manual_seed_all(args.seed)
class Proposed_prune():
    def __init__(self, args) -> None:
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args

        data = Data(args.seed)
        train_dataset, val_dataset, testdataset = data.get_dataset(dataset=args.dataset)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
        #train_loader = cycle(train_loader)
        self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=True)
        if args.arch_type == "fc1":
            self.model = fc1.fc1().to(self.device)
        elif args.arch_type == "lenet5":
            self.model = LeNet5.LeNet5().to(self.device)
        elif args.arch_type == "alexnet":
            self.model = AlexNet.AlexNet().to(self.device)
            args.lr = 0.01
            args.weight_decay = 0.0005
            args.momentum = 0.9
        elif args.arch_type == "vgg16":
            self.model = vgg.vgg16_bn(num_classes=10).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
        elif args.arch_type == "resnet18":
            self.model = resnet.resnet18().to(self.device)   
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
        elif args.arch_type == "densenet121":
            self.model = densenet.densenet121().to(self.device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()
        self.parameters_to_prune = []
        self.initial_num_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if name == 'conv1' or name == 'features.0':
                    if args.prune_conv1:
                        self.parameters_to_prune.append((module, 'weight'))
                    else:
                        print('skip the first conv2d for L1 unstructure global pruning')
                else:
                    self.parameters_to_prune.append((module, 'weight'))
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.initial_num_weights.append(param.numel())
        self.initial_num_weights = np.array(self.initial_num_weights)
        self.parameters_to_prune = tuple(self.parameters_to_prune)
        self.iter_per_epoch = int(np.ceil(len(train_dataset) / args.batch_size))
        self.total_iter = self.iter_per_epoch * args.train_epochs
        if args.prune_type == 'local':
            for layer, name in self.parameters_to_prune:
                prune.random_unstructured(layer, name=name, amount=1.0-args.initial_percent*0.01)
                self.alpha_a = np.log(args.target_ratio*0.01)
                self.alpha_b = np.log(args.output_target_ratio*0.01)
        elif args.prune_type == 'global':
            try: 
                prune.random_unstructured(self.model.conv1, 'weight', amount=0.)
            except:
                prune.random_unstructured(self.model.features[0], 'weight', amount=0.)
            prune.global_unstructured(self.parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=1.0-args.initial_percent*0.01)
            self.alpha = np.log(args.target_ratio*0.01)
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
        writer = SummaryWriter(self.save_path)
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
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
        flops_ = np.zeros(args.train_epochs, float)
        step = 0
        all_loss = np.zeros(args.train_epochs,float)
        valacc = np.zeros(args.train_epochs,float)
        early_stop_trigger = 0
        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, 0)
        sparsity = round(100.0-comp1, 1)
        sparsity_[0] = sparsity
        comp[0] = comp1
        pbar = tqdm(range(args.train_epochs))
        flops_total = 0
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
            
            if 'vgg' in self.args.arch_type:
                if train_epoch + 1 == 10 or train_epoch + 1 == 80 or train_epoch + 1 == 120:
                    for g in optimizer.param_groups:
                        g['lr'] /= 10
            # Training
            loss, flops_epoch = self.train(self.model, self.train_loader, optimizer, criterion, train_epoch, self.args)
            flops_total += flops_epoch
            flops_[train_epoch] = flops_total
            torch.save(self.model, os.path.join(self.save_path, f"{train_epoch}_model_{args.prune_type}.pt"))
            all_loss[train_epoch] = loss
            valacc[train_epoch] = val_accuracy
            # Frequency for Printing Accuracy and Loss
            if train_epoch % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} FLOPs: {flops_epoch} Loss: {loss:.6f} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            if args.early_stop is not None and early_stop_trigger > args.early_stop:
                break

            comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, train_epoch)
            sparsity_[train_epoch] = round(100.0-comp1, 1)
            test_accuracy = self.test(self.model, self.test_loader, criterion)
            print(f'Test Accuracy: {test_accuracy}')
            writer.add_scalar('Accuracy_sparsity/val', best_accuracy, sparsity)
            writer.add_scalar('Accuracy_sparsity/test', test_accuracy, sparsity)
            writer.add_scalar('Accuracy_epoch/test', test_accuracy, train_epoch)
            bestacc[0] = best_accuracy
            testacc[train_epoch] = test_accuracy
            # if train_epoch > 4:
            fig_test = utils.plot_sparsity_testacc(sparsity_[10:train_epoch+1], testacc[10:train_epoch+1], self.plot_path, name='test')
            fig_val = utils.plot_sparsity_testacc(sparsity_[10:train_epoch+1], valacc[10:train_epoch+1], self.plot_path, name='val')
            writer.add_figure('sparsity_testacc', fig_test, train_epoch)
            writer.add_figure('sparsity_valacc', fig_val, train_epoch)
            d = {'sparsity': sparsity_[: train_epoch+1], 'testacc': testacc[:train_epoch+1], 'flops': flops_[:train_epoch+1]}
            df = pd.DataFrame(data=d)
            df.to_csv(f"{self.save_path}/sparsity_vs_testacc.csv")

        # for name, p in self.model.named_parameters():
        #     weight_nz = p[torch.nonzero(p, as_tuple=True)]
        #     plt.hist(weight_nz.cpu().data.view(-1), bins=30)
        #     plt.savefig(os.path.join(self.plot_path, f"{name}.png"))
        #     plt.close()



    def train(self, model, train_loader, optimizer, criterion, train_epoch, args):
        metric = MeanMetric()
        EPS = 1e-6
        model.train()
        flops = 0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            train_iter = train_epoch * self.iter_per_epoch + batch_idx
            optimizer.zero_grad()
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            output = model(imgs)
            b, c, h, w = imgs.shape
            flops_temp, _ = get_model_complexity_info(self.model, (b, c, h, w), as_strings=False, print_per_layer_stat=False, verbose=False)
            flops += flops_temp
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
            self.prune_and_reconnect(model, args.prune_prob, args.prune_ratio * 0.01, train_iter)
        return metric.compute().item(), flops

    def test(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy
    
    def prune_and_reconnect(self, model, prune_prob, prune_ratio, train_iter):
        if random.uniform(0, 1) < prune_prob:
            mask_bef_pru = []
            prune_num_iter = []
            already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
            # prune the connections
            if args.prune_type == 'local':
                prune_ratio_a = self.exp_pruning_schedule(train_iter, self.alpha_a)
                prune_ratio_b = self.exp_pruning_schedule(train_iter, self.alpha_b)
            
                for i in range(len(self.parameters_to_prune)):
                    if not self.args.prune_conv1:
                        if i == 0:
                            prune_num_iter.append(0.)
                        elif i != len(self.parameters_to_prune) - 1:
                            prune_num_iter.append(self.initial_num_weights[i] * prune_ratio_a)
                        else:
                            prune_num_iter.append(self.initial_num_weights[i] * prune_ratio_b)
                    else: 
                        if i != len(self.parameters_to_prune) - 1:
                            prune_num_iter.append(self.initial_num_weights[i] * prune_ratio_a)
                        else:
                            prune_num_iter.append(self.initial_num_weights[i] * prune_ratio_b)
                prune_num_iter = np.array(prune_num_iter)
                to_prune = prune_num_iter - already_pruned 
                to_prune = np.array([int(np.floor(a)) for a in to_prune])
                for name, param in model.named_buffers():
                    mask_bef_pru.append(param.data)
                    i = 0
                    for layer, name in self.parameters_to_prune:
                        if to_prune[i] >= 1:
                            prune.l1_unstructured(layer, name=name, amount=to_prune[i]*2)
                            # prune.remove(layer, name=name)
                        i += 1
                            # reconnect the zeroed connections
                i = 0
                for module in list(model.children()):
                    mask = mask_bef_pru[i]
                    if to_prune[i] >= 1:
                        num_reconnect = to_prune[i] # add less connection than prune
                        zero_indices = (mask.view(-1) == 0).nonzero()
                    #     # shuffle it
                        idx = torch.randperm(zero_indices.nelement())
                        zero_indices = zero_indices.view(-1)[idx].view(zero_indices.size())[0:num_reconnect]
                        
                        (list(module.named_buffers())[0][1]).view(-1)[zero_indices] = 1.
                    i += 1
            elif args.prune_type == 'global':
                prune_ratio = self.exp_pruning_schedule(train_iter, self.alpha)
                # print(prune_ratio)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                # print(to_prune, self.initial_num_weights.sum()*prune_ratio, already_pruned.sum())
                global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=Prune_and_Reconnect, amount=to_prune)
                # prune.global_unstructured(self.parameters_to_prune, pruning_method=Reconnect, amount=to_prune*2)
                # for module, name in self.parameters_to_prune:
                #     prune.remove(module, name)
    
    def exp_pruning_schedule(self, cur_iter, alpha):
        return 1. - np.exp(alpha * cur_iter / self.total_iter)

def main():
    proposed_prune = Proposed_prune(args)
    proposed_prune.prune()
if __name__ == "__main__":
    main()

        

