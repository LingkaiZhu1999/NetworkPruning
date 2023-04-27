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
from archs.cifar10 import VGG16, AlexNet, LeNet5, fc1, resnet, densenet, vgg, wide_resnet
from prune_and_reconnect import Prune_and_Reconnect, prune_and_connect, Add, Prune_and_Reconnect_with_different_criteria
import global_unstructure
import global_unstructure_double_importance_scores
import pandas as pd
# from fvcore.nn import FlopCountAnalysis
# from ptflops import get_model_complexity_info
# from ptflops import pytorch_ops
import copy
# from torchstat import stat
sns.set_style('darkgrid')

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="proposed_exp_when_to_prune_and_add_with_different_criteria", help="the proposed method")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--train_epochs", default=160, type=int)
parser.add_argument("--fine_tune", default=0, help="fine tune the model after pruning and adding-back")
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--early_stop", default=None, type=int)
parser.add_argument("--prune_type", default="global", type=str, help="local | global")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="vgg16", type=str, help="fc1 | advanced_dropout_fc| lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--initial_percent", default=100, type=float, help='percentage of the weights that is trainable and initialized')
parser.add_argument("--prune_prob", default=None, type=float, help="probability to prune during train")
parser.add_argument("--target_ratio", default=5, type=float, )
parser.add_argument("--prune_conv1", default=False)
parser.add_argument("--optimizer", default="sgd", help="adam | sgd", type=str)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight decay for adam optim")
parser.add_argument("--val_set", default=False, help="whether have a val set", type=bool)
parser.add_argument("--fixed_budget", default=False, type=bool)
parser.add_argument("--end_update_iter_ratio", default=0.8, type=float)
parser.add_argument("--seed", default=1, type=int)
# parser.add_argument("--prune_ratio", default=1, type=float, help="Prune ratio during train")
# parser.add_argument("--output_target_ratio", default=5, type=float)
args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
class Proposed_prune():
    def __init__(self, args) -> None:
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args

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
        elif args.arch_type == "resnet50":
            self.model = resnet.ResNet50().to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0001
            args.momentum = 0.9
            args.train_epochs = 90
        elif "wideresnet" in args.arch_type:
            self.model = wide_resnet.Wide_ResNet(depth=22, widen_factor=2, dropout_rate=0, num_classes=10).to(self.device)
            args.weight_decay = 5e-4
            args.lr = 0.1
            args.train_epochs = 250
            args.batch_size = 128
            args.momentum = 0.9
        elif args.arch_type == "densenet121":
            self.model = densenet.densenet121().to(self.device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()
        ### load data ###
        data = Data(args.seed)
        if args.val_set == True:
            train_dataset, val_dataset, testdataset = data.get_dataset(dataset=args.dataset, val=args.val_set)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=False)
        else:
            train_dataset, testdataset = data.get_dataset(dataset=args.dataset, val=args.val_set)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=False)
        self.parameters_to_prune = []
        self.importance_scores = {}
        self.initial_num_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if name == 'conv1' or name == 'features.0':
                    if args.prune_conv1:
                        self.parameters_to_prune.append((module, 'weight'))
                        # self.importance_scores.update({(module, 'weight'): param.grad.data})
                    else:
                        print('skip the first conv2d for L1 unstructure global pruning')
                else:
                    self.parameters_to_prune.append((module, 'weight'))
                    # self.importance_scores.update({(module, 'weight'): param.grad.data})
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
                self.alpha = np.log(args.target_ratio*0.01)
                # self.alpha_b = np.log(args.output_target_ratio*0.01)
        elif args.prune_type == 'global':
            try: 
                prune.random_unstructured(self.model.conv1, 'weight', amount=0.)
            except:
                prune.random_unstructured(self.model.features[0], 'weight', amount=0.)
            prune.global_unstructured(self.parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=1.0-args.initial_percent*0.01)
            self.alpha = np.log(args.target_ratio*0.01)
        else:
            raise Exception('Invalid prune type.')
        
        # for module, _ in self.parameters_to_prune:
        #     self.importance_scores.update({(module, 'weight'): module.weight_orig.grad})

        self.save_path = f"{os.getcwd()}/saves/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_percent_{args.initial_percent}_prob_{args.prune_prob}_{args.prune_type}_target_ratio_{args.target_ratio}/{args.dataset}/"
        self.plot_path = f"{os.getcwd()}/plots/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_percent_{args.initial_percent}_prob_{args.prune_prob}_{args.prune_type}_target_ratio_{args.target_ratio}/{args.dataset}/"
        utils.checkdir(self.save_path)
        utils.checkdir(self.plot_path)
        with open(os.path.join(self.save_path, "args.txt"), 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f) 
        print('Config -----')
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)))
        print('------------')
    
    def prune(self, ):
        writer = SummaryWriter(self.save_path)
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise Exception('wrong optimizer, has to be adam or sgd')
        # for name, param in self.model.named_parameters():
        #     print(name, param.size())
        bestacc = 0.0
        best_accuracy = 0
        train_epochs = args.train_epochs + args.fine_tune
        comp = np.zeros(train_epochs,float)
        bestacc = np.zeros(train_epochs,float)
        testacc = np.zeros(train_epochs, float)
        sparsity_ = np.zeros(train_epochs, float)
        flops_ = np.zeros(train_epochs, float)
        step = 0
        all_loss = np.zeros(train_epochs,float)
        valacc = np.zeros(train_epochs,float)
        early_stop_trigger = 0
        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, 0)
        sparsity = round(100.0-comp1, 1)
        sparsity_[0] = sparsity
        comp[0] = comp1
        pbar = tqdm(range(args.train_epochs + args.fine_tune))
        flops_total = 0
        for train_epoch in pbar:
            # Frequency for Testing
            if (train_epoch % args.valid_freq == 0) and (self.args.val_set == True):
                val_accuracy = self.test(self.model, self.val_loader, criterion)

                # Save Weights
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    early_stop_trigger = 0
                    torch.save(self.model, os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                else:
                    early_stop_trigger += 1
                valacc[train_epoch] = val_accuracy
            
            if 'vgg' in self.args.arch_type:
                if (train_epoch + 1) == 10 or (train_epoch + 1) == 80 or (train_epoch + 1) == 120:
                    for g in optimizer.param_groups:
                        g['lr'] /= 10
            if 'resnet' in self.args.arch_type:
                    if (train_epoch + 1) % 30 == 0:
                        for g in optimizer.param_groups:
                            g['lr'] /= 10
            # Training
            loss, flops_epoch = self.train(self.model, self.train_loader, optimizer, criterion, train_epoch, self.args)
            flops_total += flops_epoch
            flops_[train_epoch] = flops_total
            all_loss[train_epoch] = loss
            # Frequency for Printing Accuracy and Loss
            if (train_epoch % args.print_freq == 0) and (self.args.val_set == True):
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} FLOPs: {flops_epoch} Loss: {loss:.6f} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            else:
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} FLOPs: {flops_epoch} Loss: {loss:.6f}')
            if args.early_stop is not None and early_stop_trigger > args.early_stop:
                break

            comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, train_epoch)
            sparsity_[train_epoch] = round(100.0 - comp1, 1)
            # if self.args.val_set == False:
            #     test_accuracy = self.test(self.model, self.test_loader, criterion)
            # else:
            #     if self.args.fixed_budget == True:
            #         best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
            #         test_accuracy = self.test(best_val_model, self.test_loader, criterion)
            #     else:
            
            if self.args.val_set == False and self.args.fixed_budget == True:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            if self.args.val_set == False and self.args.fixed_budget == False:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            if self.args.val_set == True and self.args.fixed_budget == True:
                best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                test_accuracy = self.test(best_val_model, self.test_loader, criterion)
            if self.args.val_set == True and self.args.fixed_budget == False:
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
            torch.cuda.empty_cache()
        torch.save(self.model, os.path.join(self.save_path, f"final_model_{args.prune_type}.pt"))

        # for name, p in self.model.named_parameters():
        #     weight_nz = p[torch.nonzero(p, as_tuple=True)]
        #     plt.hist(weight_nz.cpu().data.view(-1), bins=30)
        #     plt.savefig(os.path.join(self.plot_path, f"{name}.png"))
        #     plt.close()



    def train(self, model, train_loader, optimizer, criterion, train_epoch, args):
        metric = MeanMetric()
        model.train()
        flops = 0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            train_iter = train_epoch * self.iter_per_epoch + batch_idx + 1
            # if train_iter % 30000 == 0 and 'wideresnet' in self.args.arch_type:
            #     for g in optimizer.param_groups:
            #         g['lr'] /= 5
                    # args.prune_prob /= 5

            # with torch.no_grad():
            # b, c, h, w = imgs.shape
            # flops_temp, _ = get_model_complexity_info(model, (b, c, h, w), as_strings=False, print_per_layer_stat=False, verbose=False)
            # flops += flops_temp
            # del model.__dict__["start_flops_count"], model.__dict__["stop_flops_count"], model.__dict__["reset_flops_count"], model.__dict__["compute_average_flops_cost"]
            optimizer.zero_grad()
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            imgs.requires_grad = True
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            metric.update(train_loss.detach().cpu())
            # for param, _ in self.parameters_to_prune:
            #     self.importance_scores.update({param: param.weight})
            for module, _ in self.parameters_to_prune:
                self.importance_scores.update({(module, 'weight'): module.weight_orig.grad})
            optimizer.step()
            # if train_epoch < self.args.train_epochs:
            if train_iter <= int(self.args.end_update_iter_ratio * self.total_iter):
                self.prune_and_reconnect(model, args.prune_prob, train_iter)
            else:
                pass
        ### run for each epoch ###
        # self.prune_and_reconnect(model, self.args.prune_prob, train_iter)
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
    
    def prune_and_reconnect(self, model, prune_prob, train_iter):
        if random.uniform(0, 1) < self.when_to_prune(iter=train_iter):
            mask_bef_pru = []
            prune_num_iter = []
            already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
            # prune the connections
            if args.prune_type == 'local':
                if self.args.fixed_budget == False:
                    expected_ratio = 1. - self.expected_ratio_schedule(train_iter, self.alpha)
                    prune_ratio = 1. - self.pruning_schedule(train_iter, self.alpha - 1)
                    add_ratio = prune_ratio - expected_ratio
                    for module, name in self.parameters_to_prune:
                        prune_and_connect(module, name, amount_prune=prune_ratio, amount_add=add_ratio)
                else:
                    prune_ratio = 0.5*(1. - self.pruning_schedule(train_iter, self.alpha))
                    for module, name in self.parameters_to_prune:
                        prune_and_connect(module, name, amount_prune=prune_ratio, amount_add=prune_ratio)
            elif args.prune_type == 'global':
                if self.args.fixed_budget == False:
                    prune_ratio = 1. - self.pruning_schedule_exp(train_iter, self.alpha)
                    to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_and_Reconnect_with_different_criteria, amount_prune=to_prune*2, amount_add=to_prune, importance_scores_prune=None, importance_scores_add=self.importance_scores)
                else:
                    prune_ratio = 0.5*(1. - self.pruning_schedule(train_iter, self.alpha))
                    global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=Prune_and_Reconnect, amount_prune=prune_ratio, amount_add=prune_ratio)
    
    def expected_ratio_schedule(self, cur_iter, alpha=0):
        return np.exp(alpha * cur_iter / self.total_iter)
        # return np.cos(np.arccos(self.args.target_ratio*0.01)*cur_iter/self.total_iter)
    
    def pruning_schedule_exp(self, cur_iter, alpha=0):
        return np.exp(alpha * cur_iter / int(self.total_iter * self.args.end_update_iter_ratio))
        # return 0.99*np.cos(np.arccos(self.args.target_ratio*0.01)*cur_iter/self.total_iter)
    
    def pruning_schedule_cos(self, cur_iter, alpha=0):
        self.alpha = np.arccos(2 * (self.args.target_ratio * 0.01 - 0.5))
        return 0.5 * (1 + np.cos(self.alpha*cur_iter/self.total_iter))
    def exp_adding_schedule(self, cur_iter, alpha=-1):
        return 1. - np.exp(alpha * cur_iter / self.total_iter)
    
    def exp_annealing_prob(self, cur_iter):
        return 0.01 * np.exp(- cur_iter / self.total_iter)

    def when_to_prune(self, iter):
        if self.args.prune_prob is None:
            return 1/2*(1+np.cos(np.pi*iter/self.total_iter))
            # alpha = np.arccos(2 * (0.005 - 0.5))
            # return 1/2*(1+np.cos(alpha*iter/self.total_iter))
        else:
            return self.args.prune_prob

def main():
    proposed_prune = Proposed_prune(args)
    proposed_prune.prune()
if __name__ == "__main__":
    main()

        

