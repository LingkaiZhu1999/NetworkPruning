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
import utils
import random
import matplotlib.pyplot as plt
from data import Data
from archs.cifar10 import VGG16, AlexNet, LeNet5, fc1, resnet, densenet, vgg, wide_resnet
from prune_and_reconnect import Prune_and_Reconnect, prune_and_connect, Prune_and_Reconnect_with_different_criteria, Prune_and_Reconnect_with_multiple_criteria,\
Prune_GradfromW_Add_Grad, Prune_rankW_add_rankGrad_Add_Grad, Prune_rankW_add_rankGrad_Add_Random, Prune_rankW_mul_rankGrad_Add_Random, Prune_rankW_mul_rankGrad_Add_Grad, Prune_rankW_add_rankGrad, Prune_WfromGrad_Add_Grad, Prune_WfromGrad_Add_Grad_and_Random
from prune_and_reconect1 import prune_WfromGrad_Add_Grad
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
parser.add_argument("--method", default="proposed_exp_when_to_prune", help="the proposed method")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--train-epochs", default=160, type=int)
parser.add_argument("--fine-tune", default=0, help="fine tune the model after pruning and adding-back")
parser.add_argument("--print-freq", default=1, type=int)
parser.add_argument("--valid-freq", default=1, type=int)
parser.add_argument("--early-stop", default=None, type=int)
parser.add_argument("--prune-type", default="global", type=str, help="local | global")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--dataset", default="cifar100", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch-type", default="vgg16", type=str, help="fc1 | advanced_dropout_fc| lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--initial-ratio", default=100, type=float, help='percentage of the weights that is trainable and initialized')
parser.add_argument("--target-ratio", default=5, type=float, )
parser.add_argument("--prune-rate", default=1, type=float)
parser.add_argument("--prune-conv1", default=False, type=bool)
parser.add_argument("--optimizer", default="sgd", help="adam | sgd", type=str)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=0.0005, type=float, help="weight decay for adam optim")
parser.add_argument("--val-set", default=False, help="whether have a val set", type=bool)
parser.add_argument("--fixed-budget", default=False, type=bool)
parser.add_argument("--end-update-iter-ratio", default=0.8, type=float)
parser.add_argument("--prune-criterion", default="", type=str)
parser.add_argument("--add-criterion", default="", type=str)
parser.add_argument("--schedule-function", default='cubic', type=str, help='exp or cubic scheduling functions')
parser.add_argument("--moving-average-alpha", default=0.5, type=float)
parser.add_argument("--update-interval", type=int, default=2000)
parser.add_argument("--init-type", type=str, default="")
parser.add_argument("--select-ratio", type=float, default=0.5)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
class Proposed_prune():
    def __init__(self, args) -> None:
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args
        self.counters = 1
        if self.args.dataset == "cifar10":
            num_classes = 10
        elif self.args.dataset == "cifar100":
            num_classes = 100
        elif self.args.dataset == "imagenet":
            num_classes = 1000
        elif self.args.dataset == "tinyimagenet":
            num_classes = 200
        if args.arch_type == "fc":
            self.model = fc1.fc1().to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.batch_size = 128
            args.momentum = 0.9
            args.train_epochs = 160
        elif args.arch_type == "lenet5":
            self.model = LeNet5.LeNet5().to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.batch_size = 128
            args.momentum = 0.9
            args.train_epochs = 160
        elif args.arch_type == "alexnet":
            self.model = AlexNet.AlexNet().to(self.device)
            args.lr = 0.01
            args.weight_decay = 0.0005
            args.momentum = 0.9
        elif args.arch_type == "vgg16":
            self.model = vgg.vgg16_bn(num_classes=num_classes).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.momentum = 0.9
        elif args.arch_type == "vgg19":
            self.model = vgg.vgg19_bn(num_classes=num_classes).to(self.device)
            # self.model = vgg.VGG(depth=19, dataset=args.dataset, batchnorm=True).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.momentum = 0.9
        elif args.arch_type == "resnet50":
            # self.model = resnet.ResNet50(num_classes=num_classes).to(self.device)
            self.model = resnet.resnet(depth=50, dataset=args.dataset).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.momentum = 0.9
        elif args.arch_type == "resnet32":
            self.model = resnet.resnet(depth=32, dataset=args.dataset).to(self.device)
            args.lr = 0.1
            args.weight_decay = 0.0005
            args.momentum = 0.9
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
        
        self.prune_rate_decay = CosineDecay(prune_rate=args.prune_rate*0.01, T_max=len(self.train_loader)*self.train_epochs)
        self.add_rate_decay = CosineDecay(prune_rate=1., T_max=len(self.train_loader)*self.train_epochs)
        self.parameters_to_prune = []
        self.initial_num_weights = []
        self.importance_scores_prune = {} 
        self.importance_scores_add = {}
        if self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)" or self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
            self.importance_scores_prune1 = {} 
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if name == 'conv1' or name == 'features.0':
                    if args.prune_conv1:
                        self.parameters_to_prune.append((module, 'weight'))
                    else:
                        print('skip the first conv2d for L1 unstructure global pruning')
                else:
                    self.parameters_to_prune.append((module, 'weight'))
        # if 'fc' in self.args.arch_type or 'lenet5' in self.args.arch_type:
        #     for name, param in self.model.named_parameters():
        #         if 'weight' in name or 'bias' in name:
        #             self.initial_num_weights.append(param.numel())
        # elif "vgg" in self.args.arch_type:
        #     for name, param in self.model.named_parameters():
        #         if 'weight' in name:
        #             self.initial_num_weights.append(param.numel())
        # else:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.initial_num_weights.append(module.weight.numel())
                if module.bias is not None:
                    self.initial_num_weights.append(module.bias.numel())
        self.initial_num_weights = np.array(self.initial_num_weights)
        print(self.initial_num_weights)
        self.parameters_to_prune = tuple(self.parameters_to_prune)
        self.iter_per_epoch = len(self.train_loader)
        print("iteration per epoch: ", self.iter_per_epoch)
        self.total_iter = len(self.train_loader) * args.train_epochs
        if args.prune_type == 'local':
            for layer, name in self.parameters_to_prune:
                prune.random_unstructured(layer, name=name, amount=0)
        elif args.prune_type == 'global':
            try: 
                prune.random_unstructured(self.model.conv1, 'weight', amount=0.)
            except:
                if self.args.arch_type == 'fc':
                    prune.random_unstructured(self.model.fc1, 'weight', amount=0.)
                else:
                    prune.random_unstructured(self.model.features[0], 'weight', amount=0.)
            prune.global_unstructured(self.parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=0)
            self.alpha = np.log((args.target_ratio * 0.01) / (args.initial_ratio * 0.01))
        else:
            raise Exception('Invalid prune type.')

        if self.args.init_type == "ERK":
            # adapted from GraNet's code https://github.com/VITA-Group/GraNet 
            is_epsilon_valid = False
            erk_power_scale = 1.0
            dense_layers = set()

            while not is_epsilon_valid:

                divisor = 0
                rhs =  0
                raw_probabilities = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        n_param = np.prod(module.weight.shape)
                        n_zeros = n_param * (1 - self.args.initial_ratio * 0.01)
                        n_ones = n_param * self.args.initial_ratio * 0.01

                        if name in dense_layers:
                            rhs -= n_zeros

                        else:
                            rhs += n_ones
                            raw_probabilities[name] = (
                                np.sum(module.weight.shape) / np.prod(module.weight.shape)
                            ) ** erk_power_scale
                            divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True
            density_dict = {}
            total_nonzero = 0.0
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    n_param = np.prod(module.weight_mask.shape)
                    if name in dense_layers:
                        density_dict[name] = 1.0
                    else:
                        probability_one = epsilon * raw_probabilities[name]
                        density_dict[name] = probability_one
                    print(
                            f"layer: {name}, shape: {module.weight_mask.shape}, nnz_ratio: {density_dict[name]}"
                        )
                    module.weight_mask = (torch.rand(module.weight_mask.shape) < density_dict[name]).float().data.to(self.args.device)
                    total_nonzero += density_dict[name] * module.weight_mask.numel()

            print(f"Overall sparsity {total_nonzero / self.initial_num_weights.sum()}")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.weight *= module.weight_mask

        
        self.save_path = f"{os.getcwd()}/saves/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_ratio_{args.initial_ratio}_{args.prune_type}_target_ratio_{args.target_ratio}_end_update_iter_ratio_{args.end_update_iter_ratio}_prune_rate_{args.prune_rate}_seed_{args.seed}/{args.dataset}/"
        self.plot_path = f"{os.getcwd()}/plots/{args.method}/{args.dataset}/{args.arch_type}_lr_{args.lr}_{args.optimizer}_initial_ratio_{args.initial_ratio}_{args.prune_type}_target_ratio_{args.target_ratio}_end_update_iter_ratio_{args.end_update_iter_ratio}_prune_rate_{args.prune_rate}_seed_{args.seed}/{args.dataset}/"
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
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        else:
            raise Exception('wrong optimizer, has to be adam or sgd')
        # if 'vgg' in self.args.arch_type or 'resnet' in self.args.arch_type or 'lenet' in self.args.arch_type:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epochs / 2), int(args.train_epochs * 3 / 4)], last_epoch=-1)
        # else:
            # lr_scheduler = None
        bestacc = 0.0
        best_accuracy = 0
        test_accuracy = 0.0
        train_epochs = args.train_epochs
        comp = np.zeros(train_epochs,float)
        bestacc = np.zeros(train_epochs,float)
        testacc = np.zeros(train_epochs, float)
        sparsity_ = np.zeros(train_epochs, float)
        all_loss = np.zeros(train_epochs,float)
        valacc = np.zeros(train_epochs,float)
        early_stop_trigger = 0
        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, 0)
        sparsity = round(100.0-comp1, 1)
        sparsity_[0] = sparsity
        comp[0] = comp1
        pbar = tqdm(range(args.train_epochs))
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
            
            # Training
            loss = self.train(self.model, self.train_loader, optimizer, criterion, train_epoch, self.args)
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            all_loss[train_epoch] = loss
            # Frequency for Printing Accuracy and Loss
            if (train_epoch % args.print_freq == 0) and (self.args.val_set == True):
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} Loss: {loss:.6f} Prune rate: {self.prune_rate_decay.get_dr()} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            else:
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} Loss: {loss:.6f} Prune rate: {self.prune_rate_decay.get_dr()} Add rate: {self.add_rate_decay.get_dr()}')
            if args.early_stop is not None and early_stop_trigger > args.early_stop:
                break
            comp1 = utils.print_nonzeros_lth(self.model.named_modules(), writer, train_epoch)
            sparsity_[train_epoch] = round(100.0 - comp1, 1)
            
            if self.args.val_set == False and self.args.fixed_budget == True:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            if self.args.val_set == False and self.args.fixed_budget == False:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            if self.args.val_set == True and self.args.fixed_budget == True:
                best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                test_accuracy = self.test(best_val_model, self.test_loader, criterion)
            if self.args.val_set == True and self.args.fixed_budget == False:
                if train_epoch > int(np.ceil(self.args.end_update_iter_ratio * self.args.train_epochs)):
                    best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                    test_accuracy = self.test(best_val_model, self.test_loader, criterion)
                else:
                    test_accuracy = self.test(self.model, self.test_loader, criterion)

            print(f'Test Accuracy: {test_accuracy}')
            writer.add_scalar('Accuracy_sparsity/val', best_accuracy, sparsity)
            writer.add_scalar('Accuracy_sparsity/test', test_accuracy, sparsity)
            writer.add_scalar('Accuracy_epoch/test', test_accuracy, train_epoch)
            bestacc[0] = best_accuracy
            testacc[train_epoch] = test_accuracy
            fig_test = utils.plot_sparsity_testacc(sparsity_[20:train_epoch+1], testacc[20:train_epoch+1], self.plot_path, name='test')
            fig_val = utils.plot_sparsity_testacc(sparsity_[20:train_epoch+1], valacc[20:train_epoch+1], self.plot_path, name='val')
            writer.add_figure('sparsity_testacc', fig_test, train_epoch)
            writer.add_figure('sparsity_valacc', fig_val, train_epoch)
            d = {'sparsity': sparsity_[: train_epoch+1], 'testacc': testacc[:train_epoch+1]}
            df = pd.DataFrame(data=d)
            df.to_csv(f"{self.save_path}/sparsity_vs_testacc.csv")
            torch.cuda.empty_cache()
        # torch.save(self.model, os.path.join(self.save_path, f"final_model_{args.prune_type}.pt"))
        for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune.remove(module, 'weight')
        save_checkpoint({
        'epoch': (train_epoch + 1),
        'state_dict': self.model.state_dict(),
        'final_acc': test_accuracy,
        'optimizer': optimizer.state_dict(),
            }, self.save_path + 'model_final.pth.tar')



    def train(self, model, train_loader, optimizer, criterion, train_epoch, args):
        metric = MeanMetric()
        model.train()
        if len(self.args.prune_criterion) > 0:
            if self.args.prune_criterion == "average w":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): module.weight})
            elif self.args.prune_criterion == "average grad":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): module.weight_orig.grad})
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            train_iter = train_epoch * len(self.train_loader) + batch_idx + 1
            optimizer.zero_grad()
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            metric.update(train_loss.detach().cpu())
            if self.args.prune_criterion == "|gradw|" and (train_iter % self.args.update_interval == 0):
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): module.weight_orig.grad * module.weight})
            elif self.args.prune_criterion == "|w|" and (train_iter % self.args.update_interval == 0):
                self.importance_scores_prune = None
            elif self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)" or self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune1.update({(module, 'weight'): module.weight_orig.grad})
            elif self.args.prune_criterion == "|grad|":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): module.weight_orig.grad})
            elif self.args.prune_criterion == "average w":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): self.args.moving_average_alpha * self.importance_scores_prune.get((module, 'weight')) + (1 - self.args.moving_average_alpha) * module.weight})
            elif self.args.prune_criterion == "average grad":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_prune.update({(module, 'weight'): self.args.moving_average_alpha * self.importance_scores_prune.get((module, 'weight')) + (1 - self.args.moving_average_alpha) * module.weight_orig.grad})
            # elif self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
            #     for module, _ in self.parameters_to_prune:
            #         self.importance_scores_prune.update({(module, 'weight'): torch.abs(module.weight).sort(descending=False).indices + torch.abs(module.weight_orig.grad).sort(descending=False).indices})
                # if (batch_idx + 1 == int(0.1 * self.iter_per_epoch + 1)) or (batch_idx + 1 == int(0.3 * self.iter_per_epoch + 1)) or (batch_idx + 1 == int(0.5 * self.iter_per_epoch + 1)) or (batch_idx + 1 == int(0.7 * self.iter_per_epoch + 1)) or (batch_idx + 1 == int(0.9 * self.iter_per_epoch + 1)):
                #     for module, _ in self.parameters_to_prune:
                #         self.importance_scores_prune.update({(module, 'weight'): module.weight})
            if self.args.add_criterion == "|grad|":
                for module, _ in self.parameters_to_prune:
                    self.importance_scores_add.update({(module, 'weight'): module.weight_orig.grad})
            elif self.args.add_criterion == "random" or self.args.add_criterion == "|w|" or self.args.add_criterion == "|w| + random":
                self.importance_scores_add = None
            self.prune_rate_decay.step()
            self.add_rate_decay.step()
            optimizer.step() 
            if train_iter <= int(self.args.end_update_iter_ratio * self.total_iter):
                self.prune_and_reconnect(train_iter)
            # else:
                # pass
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
    
    def prune_and_reconnect(self, train_iter):
        if train_iter % self.args.update_interval == 0:
            if self.args.add_criterion == "random":
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune.remove(module, 'weight')
                prune_ratio = 1. - self.schedule_function(train_iter)
                add_ratio = self.prune_rate_decay.get_dr() * self.schedule_function(train_iter)
                if self.args.prune_criterion == "|w|":
                    global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=Prune_and_Reconnect, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores=self.importance_scores_prune)
                elif self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_add_rankGrad, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)
                elif self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_mul_rankGrad_Add_Random, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)


            elif self.args.add_criterion == "|grad|":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                remain = self.initial_num_weights.sum() - already_pruned.sum()
                to_prune_t = to_prune
                to_add_t = int((remain - to_prune_t) * self.prune_rate_decay.get_dr())
                
                # modification 1, exchange w and grad
                # if train_iter > int(self.args.end_update_iter_ratio * self.total_iter):
                #     to_prune = 0
                # remain = self.initial_num_weights.sum() - already_pruned.sum()
                # to_add_t = int(args.select_ratio * remain) - to_prune
                # to_prune_t = int(args.select_ratio * remain)
                ################################################################

                if self.args.prune_criterion == "|w|":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_GradfromW_Add_Grad, amount_prune=(to_prune_t+to_add_t), amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                    #global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_GradfromW_Add_Grad, amount_prune=to_prune_t, amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
            elif self.args.add_criterion == "":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                if self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
                    to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                    if self.args.prune_type == "global":
                        global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_add_rankGrad, amount_prune=prune_ratio, amount_add=0, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                    else:
                        i = 0
                        for module, name in self.parameters_to_prune:
                            to_prune = int(np.floor((torch.numel(module.weight) * prune_ratio - already_pruned[i])))
                            global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(tuple([(module, 'weight')]), pruning_method=Prune_rankW_add_rankGrad, amount_prune=to_prune, amount_add=0, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                            i += 1
                if self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
                    to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                    if self.args.prune_type == "global":
                        global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_mul_rankGrad_Add_Grad, amount_prune=prune_ratio, amount_add=0, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                    else:
                        i = 0
                        for module, name in self.parameters_to_prune:
                            to_prune = int(np.floor((torch.numel(module.weight) * prune_ratio - already_pruned[i])))
                            global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(tuple([(module, 'weight')]), pruning_method=Prune_rankW_mul_rankGrad_Add_Grad, amount_prune=to_prune, amount_add=0, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                            i += 1
                if self.args.prune_criterion == "|gradw|":
                    to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                    global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=prune.L1Unstructured, amount=to_prune,importance_scores=self.importance_scores_prune)
                        
            elif self.args.add_criterion == "|w|":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                # to_add_t = int((remain - to_prune_t) * self.prune_rate_decay.get_dr())
                if self.args.prune_criterion == "|grad|":
                    if train_iter > int(self.args.end_update_iter_ratio * self.total_iter):
                        to_prune = 0
                    remain = self.initial_num_weights.sum() - already_pruned.sum()
                    to_add_t = int(self.args.select_ratio * remain) - to_prune
                    to_prune_t = int(self.args.select_ratio * remain)
                    # if to_add_t < 0:
                    #     to_add_t = int(0.8*remain) - to_prune
                    #     to_prune_t = remain
                    # modification 2 #####################################################
                    # remain = self.initial_num_weights.sum() - already_pruned.sum()
                    # to_prune_t = to_prune
                    # to_add_t = int((remain - to_prune_t) * self.prune_rate_decay.get_dr())
                    ######################################################################
                    # modification 3 #####################################################
                    # remain = self.initial_num_weights.sum() - already_pruned.sum()
                    # to_prune_t = to_prune
                    # to_add_t = int((remain - to_prune_t) * 0.5)
                    ######################################################################
                    if self.args.prune_type == "global":
                        global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_WfromGrad_Add_Grad, amount_prune=to_prune_t, amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                        # global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_WfromGrad_Add_Grad, amount_prune=(to_prune_t+to_add_t), amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add) 
                    
                    elif self.args.prune_type == "local":
                        to_prune_t = to_prune_t / self.initial_num_weights.sum()
                        to_add_t = to_add_t / self.initial_num_weights.sum()
                        for module, name in self.parameters_to_prune:
                            global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(tuple([(module, 'weight')]), pruning_method=Prune_WfromGrad_Add_Grad, amount_prune=to_prune_t, amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add) 
            elif self.args.add_criterion == "|w| inter random":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                # to_add_t = int((remain - to_prune_t) * self.prune_rate_decay.get_dr())
                if self.args.prune_criterion == "|grad|":
                    if train_iter > int(self.args.end_update_iter_ratio * self.total_iter):
                        to_prune = 0
                    remain = self.initial_num_weights.sum() - already_pruned.sum()
                    to_add_t = int(0.5 * remain) - to_prune
                    to_prune_t = int(0.5 * remain)
                    if to_add_t < 0:
                        to_add_t = int(0.8*remain) - to_prune
                        to_prune_t = remain
                    if self.counters % 5 == 0:
                        for name, module in self.model.named_modules():
                            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                                prune.remove(module, 'weight')
                        to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio)))
                        to_prune_t = already_pruned.sum() + int(0.5 * remain)
                        to_add_t = to_prune_t - to_prune
                        global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=Prune_and_Reconnect, amount_prune=to_prune_t, amount_add=to_add_t, importance_scores=None)
                        # print("random")
                    else:
                        global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_WfromGrad_Add_Grad, amount_prune=to_prune_t, amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                    self.counters += 1
            elif self.args.add_criterion == "|w| + random":
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune.remove(module, 'weight')
                if self.args.prune_criterion == "|grad|":
                    prune_ratio = 1. - self.schedule_function(train_iter)
                    print(prune_ratio)
                    prune_number = int(np.floor(prune_ratio * self.initial_num_weights.sum()))
                    # print(prune_ratio * )
                    # if train_iter > int(self.args.end_update_iter_ratio * self.total_iter):
                    #     prune_ratio = 1 - self.args.target_ratio * 0.01
                    remain = 1 - prune_ratio
                    prune_ratio += self.prune_rate_decay.get_dr() * remain
                    to_prune = int(prune_ratio * self.initial_num_weights.sum())
                    # add_ratio_random = (1 - self.add_rate_decay.get_dr()) * self.prune_rate_decay.get_dr() * remain
                    # add_ratio_grad = self.add_rate_decay.get_dr() * self.prune_rate_decay.get_dr() * remain
                    add_ratio_random = 0.05 * self.prune_rate_decay.get_dr() * remain
                    to_add_random = int(add_ratio_random * self.initial_num_weights.sum())
                    add_ratio_grad = 0.95 * self.prune_rate_decay.get_dr() * remain
                    to_add_grad = int(add_ratio_grad * self.initial_num_weights.sum())
                    # print(prune_number, to_prune, to_add_random, to_add_grad)
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_WfromGrad_Add_Grad_and_Random, amount_prune=to_prune, amount_add_grad=to_add_grad, amount_add_random=to_add_random, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                    

            
    def schedule_function(self, cur_iter):
        if self.args.schedule_function == "exp":
            return self.args.initial_ratio * 0.01 * np.exp(self.alpha * cur_iter / int(self.total_iter * self.args.end_update_iter_ratio))
        elif self.args.schedule_function == "cubic":
            return self.args.target_ratio * 0.01 + 0.01*(self.args.initial_ratio - self.args.target_ratio)*(1 - (cur_iter/(int(self.total_iter * self.args.end_update_iter_ratio))))**3
    
class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = torch.optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)

def main():
    proposed_prune = Proposed_prune(args)
    proposed_prune.prune()
if __name__ == "__main__":
    main()

        

