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
import pandas as pd
from global_unstructure import global_unstructured
from archs.cifar10 import AlexNet, LeNet5, fc1, resnet, vgg, wide_resnet
# from ptflops import get_model_complexity_info
sns.set_style('darkgrid')
parser = argparse.ArgumentParser()
parser.add_argument("--method", default="LTH", help="name of the method, it is Lottery Ticket Hypothesis")
parser.add_argument("--lr",default=0.01, type=float, help="Learning rate") # learning rate have a big effect
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--start_prune_prune_round", default=0, type=int)
parser.add_argument("--train_epochs", default=160, type=int)
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--early_stop", default=None, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--retrain_type", default="original", type=str, help="original | reinit")
parser.add_argument("--prune_type", default="global", help="local | global")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="wideresnet", type=str, help="fc1 | advanced_dropout_fc | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
parser.add_argument("--prune_rounds", default=20, type=int, help="Pruning args.prune_roundss count")
parser.add_argument("--prune_conv1", default=False)
parser.add_argument("--optimizer", default="sgd", help="adam | sgd")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight decay for adam optim")
parser.add_argument("--val_set", default=False, help="whether have a val set")
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

class LTH():
    def __init__(self, args) -> None:
        self.device = args.device
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
        elif args.arch_type == "vgg19":
            self.model = vgg.vgg19_bn(num_classes=10).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
        elif args.arch_type == "resnet50":
            self.model = resnet.ResNet50().to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
            # args.train_epochs = 160
        elif "wideresnet" in args.arch_type:
            self.model = wide_resnet.Wide_ResNet(depth=22, widen_factor=2, dropout_rate=0.3, num_classes=10).to(self.device)
            args.weight_decay = 5e-4
            args.lr = 0.1
            args.train_epochs = 240
            args.batch_size = 128
            args.momentum = 0.9
        elif args.arch_type == "densenet121":
            self.model = densenet.densenet121().to(self.device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args
        data = Data(args.seed)
        if args.val_set:
            train_dataset, val_dataset, testdataset = data.get_dataset(dataset=args.dataset, val=args.val_set)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=False)
        else:
            train_dataset, testdataset = data.get_dataset(dataset=args.dataset, val=args.val_set)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=False)
        
        self.parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if name == 'conv1' or name == 'features.0':
                    if args.prune_conv1:
                        self.parameters_to_prune.append((module, 'weight'))
                else:
                    self.parameters_to_prune.append((module, 'weight'))
            
        self.parameters_to_prune = tuple(self.parameters_to_prune)
        self.save_path = f"{os.getcwd()}/saves/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_{args.prune_type}/{args.dataset}/"
        self.plot_path = f"{os.getcwd()}/plots/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_{args.prune_type}/{args.dataset}/"
        utils.checkdir(self.save_path)
        utils.checkdir(self.plot_path)
        with open(os.path.join(self.save_path, "args.txt"), 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f) 

        self.mask = [None] * len(self.parameters_to_prune)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        torch.save(self.model, os.path.join(self.save_path, "initial_state_dict_{args.retrain_type}.pt"))
        
        self.iter_per_epoch = int(np.ceil(len(train_dataset) / args.batch_size))
        self.total_iter = self.iter_per_epoch * self.train_epochs 
        
    def prune(self, ):
        bestacc = 0.0
        best_accuracy = 0
        comp = np.zeros(self.args.prune_rounds,float)
        bestacc = np.zeros(self.args.prune_rounds,float)
        testacc = np.zeros(self.args.prune_rounds, float)
        sparsity_ = np.zeros(self.args.prune_rounds, float)
        flops_ = np.zeros(self.args.prune_rounds, float)
        flops = 0
        step = 0
        all_loss = np.zeros(self.args.train_epochs,float)
        all_accuracy = np.zeros(self.args.train_epochs,float)
        early_stop_trigger = 0
        reinit = True if args.retrain_type=="reinit" else False
        writer = SummaryWriter(self.save_path)
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        else:
            raise Exception('wrong optimizer, has to be adam or sgd')
        
        criterion = nn.CrossEntropyLoss()
        
        for prune_round in range(self.args.start_prune_prune_round, self.args.prune_rounds):
            # imgs, _ = next(iter(self.train_loader))
            # flops_temp, _ = get_model_complexity_info(self.model, tuple(imgs.shape), as_strings=False, print_per_layer_stat=False, verbose=False)
            # flops += flops_temp * self.total_iter 
            # flops_[prune_round] = flops
            # del self.model.__dict__["start_flops_count"], self.model.__dict__["stop_flops_count"], self.model.__dict__["reset_flops_count"], self.model.__dict__["compute_average_flops_cost"]
            
            if not prune_round == 0: # don't prune for the first running prune_round, because we want the model to be well trained before we prune it.
                print(prune_round)
                self.prune_by_percentile(self.args.prune_percent * 0.01, self.parameters_to_prune)
                self.get_mask()
                if reinit:
                    # model.apply(weight_init)
                    step = 0
                    for name, param in self.model.named_parameters():
                        if 'weight' in name:
                            param.data = (param.data * self.mask[step]).to(self.device)
                            step = step + 1
                    step = 0
                else:
                    self.original_initialization(self.mask, self.initial_state_dict)
                if args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
                elif args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
                else:
                    raise Exception('wrong optimizer, has to be adam or sgd')
            print(f"\n--- Pruning Level [{prune_round}/{self.args.prune_rounds}]: ---")
            # Print the table of Nonzeros in each layer
            comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, prune_round)
            sparsity = round(float(100.0 - comp1), 1)
            sparsity_[prune_round] = sparsity
            comp[prune_round] = comp1
            pbar = tqdm(range(self.args.train_epochs))

            for train_epoch in pbar:

                # Frequency for Testing
                if train_epoch % self.args.valid_freq == 0 and self.args.val_set:
                    val_accuracy = self.test(self.model, self.val_loader, criterion)
                    writer.add_scalar(f'{prune_round}/valacc', val_accuracy, train_epoch)
                    # Save Weights
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        torch.save(self.model, os.path.join(self.save_path, f"{prune_round}_model_{self.args.retrain_type}.pt"))
                        early_stop_trigger = 0
                    else:
                        early_stop_trigger += 1
                    all_accuracy[train_epoch] = val_accuracy

                # Training
                if 'vgg' or 'resnet' in self.args.arch_type:
                    if train_epoch + 1 == 10 or train_epoch + 1 == 80 or train_epoch + 1 == 120:
                        for g in optimizer.param_groups:
                            g['lr'] /= 10
                # if 'resnet' in self.args.arch_type:
                #     if iter_ + 1 % 30 == 0:
                #         for g in optimizer.param_groups:
                #             g['lr'] /= 10
                loss = self.train(self.model, self.train_loader, optimizer, criterion, train_epoch)
                all_loss[train_epoch] = loss
                # Frequency for Printing Accuracy and Loss
                if train_epoch % self.args.print_freq == 0:
                    pbar.set_description(
                        f'Train Epoch: {train_epoch}/{self.args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} FLOPs: {flops} Loss: {loss:.6f}')       
                if self.args.early_stop is not None and early_stop_trigger > self.args.early_stop:
                    break
            if self.args.val_set:
                best_val_model = torch.load(os.path.join(self.save_path, f"{prune_round}_model_{self.args.retrain_type}.pt"))
                test_accuracy = self.test(best_val_model, self.test_loader, criterion)
                del best_val_model
            else:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            print(f'Test Accuracy: {test_accuracy}')
            writer.add_scalar('Accuracy/val', best_accuracy, sparsity)
            writer.add_scalar('Accuracy/test', test_accuracy, sparsity)
            bestacc[prune_round] = best_accuracy
            testacc[prune_round] = test_accuracy
            fig = utils.plot_sparsity_testacc(sparsity_[:prune_round+1], testacc[:prune_round+1], self.plot_path)
            fig = utils.plot_sparsity_testacc(sparsity_[:prune_round+1], bestacc[:prune_round+1], self.plot_path, name='val')
            # save to csv
            d = {'sparsity': sparsity_[: prune_round+1], 'testacc': testacc[:prune_round+1], 'flops': flops_[:prune_round+1]}
            df = pd.DataFrame(data=d)
            df.to_csv(f"{self.save_path}/sparsity_vs_testacc.csv")

            writer.add_figure('sparsity_testacc', fig, prune_round)
            
            # Making variables into 0
            best_accuracy = 0
            all_loss = np.zeros(self.args.train_epochs,float)
            all_accuracy = np.zeros(self.args.train_epochs,float)

            torch.cuda.empty_cache()



    def train(self, model, train_loader, optimizer, criterion, train_epoch):
        metric = MeanMetric()
        EPS = 1e-6
        model.train()
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            train_iter = train_epoch * self.iter_per_epoch + batch_idx + 1
            if train_iter % 30000 == 0 and 'wideresnet' in self.args.arch_type:
                for g in optimizer.param_groups:
                    g['lr'] /= 5
                    args.prune_prob /= 5
            optimizer.zero_grad()
            #imgs, targets = next(train_loader)
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            metric.update(train_loss.detach().cpu())
            # Freezing Pruned weights by making their gradients Zero
            # for name, p in model.named_parameters():
            #     if 'weight' in name:
            #         tensor = p.data
            #         grad_tensor = p.grad.data
            #         grad_tensor = torch.where(torch.abs(tensor) < EPS, 0, grad_tensor)
            #         p.grad.data = grad_tensor.to(self.device)

            optimizer.step()
        
        return metric.compute().item()

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
    
    def prune_by_percentile(self, prune_percent, parameters_to_prune):
        if args.prune_type == 'local':
            i = 0
            for layer, name in parameters_to_prune:
                i += 1
                if i != len(parameters_to_prune):
                    prune.l1_unstructured(layer, name=name, amount=prune_percent)
                else:
                    # prune at half ratio for the last output layer
                    prune.l1_unstructured(layer, name=name, amount=prune_percent/2)
        elif args.prune_type == 'global':
            global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percent)

    def get_mask(self,):
        step = 0
        for module, _ in self.parameters_to_prune:
            self.mask[step] = list(module.named_buffers())[0][1].to(self.device).to_sparse()
            step += 1 
    
    def original_initialization(self, mask_temp, initial_state_dict):
        if args.arch_type == "fc1":
            ini_model = fc1.fc1().to(self.device)
        elif args.arch_type == "lenet5":
            ini_model = LeNet5.LeNet5().to(self.device)
        elif args.arch_type == "alexnet":
            ini_model = AlexNet.AlexNet().to(self.device)
        elif args.arch_type == "vgg16":
            ini_model = vgg.vgg16_bn(num_classes=10).to(self.device)
        elif args.arch_type == "vgg19":
            ini_model = vgg.vgg19_bn(num_classes=10).to(self.device)
        elif args.arch_type == "resnet50":
            ini_model = resnet.ResNet50().to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
            # args.train_epochs = 160
        elif "wideresnet" in args.arch_type:
            ini_model = wide_resnet.Wide_ResNet(depth=22, widen_factor=2, dropout_rate=0.3, num_classes=10).to(self.device)
            args.weight_decay = 5e-4
            args.lr = 0.1
            args.train_epochs = 1
            args.batch_size = 128
            args.momentum = 0.9
        elif args.arch_type == "densenet121":
            ini_model = densenet.densenet121().to(self.device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()
        ini_model.load_state_dict(initial_state_dict)
        # step = 0
        for module, ini_module in zip(self.model.modules(), ini_model.modules()): 
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data = (module.weight.data != 0).float() * ini_module.weight.data
                # print(torch.count_nonzero(module.weight.data == 0).int())
            # if 'conv1' in name:
            #     if self.args.prune_conv1:
            #         if "weight" in name: 
            #             param.data = (mask_temp[step] * initial_state_dict[name[:-5]]).to(self.device).to_dense()
            #             step = step + 1
            #         if "bias" in name:
            #             param.data = initial_state_dict[name]
            # else:
            #     if "weight" in name: 
            #         param.data = (mask_temp[step] * initial_state_dict[name[:-5]]).to(self.device).to_dense()
            #         step = step + 1
            #     if "bias" in name:
            #         param.data = initial_state_dict[name]
        # step = 0

    
def main():
    torch.cuda.manual_seed_all(args.seed)
    proposed_prune = LTH(args)
    proposed_prune.prune()

if __name__ == "__main__":
    main()

        

