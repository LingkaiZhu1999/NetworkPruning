# Importing Libraries
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchmetrics import MeanMetric
import os
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import torch.nn.utils.prune as prune
# Custom Libraries
import utils
from data import Data
# from archs.cifar10 import VGG16, AlexNet, LeNet5, fc1, resnet, densenet, vgg, wide_resnet
from archs.imagenet import resnet
from prune_and_reconnect import Prune_and_Reconnect, prune_and_connect, Prune_and_Reconnect_with_different_criteria, Prune_and_Reconnect_with_multiple_criteria,\
Prune_GradfromW_Add_Grad, Prune_rankW_add_rankGrad_Add_Grad, Prune_rankW_add_rankGrad_Add_Random, Prune_rankW_mul_rankGrad_Add_Random, Prune_rankW_mul_rankGrad_Add_Grad, Prune_rankW_add_rankGrad, Prune_WfromGrad_Add_Grad
import global_unstructure
import global_unstructure_double_importance_scores
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
sns.set_style('darkgrid')

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="proposed_exp_when_to_prune", help="the proposed method")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--train-epochs", default=160, type=int)
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
parser.add_argument("--update-interval", type=int, default=1000)
parser.add_argument("--init-type", type=str, default="")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--label-smoothing", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=1)
# distributed training arguments
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
args = parser.parse_args()

torch.cuda.manual_seed_all(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
class Proposed_prune():
    def __init__(self, args) -> None:
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_epochs = args.train_epochs
        self.args = args

        if args.arch_type == "resnet50":
            self.model = resnet.build_resnet(self.args.arch_type, 'fanin')
            args.lr = 0.1
            args.weight_decay = 0.0004
            args.momentum = 0.9
            args.train_epochs = 90
        
        self.parameters_to_prune = []
        self.initial_num_weights = []
        self.importance_scores_prune = {} 
        self.importance_scores_add = {}
        if self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)" or self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
            self.importance_scores_prune1 = {} 
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
    
    def prune(self, gpu, ngpus_per_node, args):
        args.gpu = gpu
        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            print('using CPU, this will be slow')
        elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
            if torch.cuda.is_available():
                if args.gpu is not None:
                    torch.cuda.set_device(args.gpu)
                    self.model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                    args.batch_size = int(args.batch_size / ngpus_per_node)
                    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
                else:
                    self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model)  
        elif args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            self.model = self.model.to(device)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                self.model.features = torch.nn.DataParallel(self.model.features)
                self.model.cuda()
            else:
                self.model = torch.nn.DataParallel(self.model).cuda()
        data = Data(args.seed)
        train_dataset, val_dataset = data.get_dataset(dataset=args.dataset, val=False)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        self.prune_rate_decay = CosineDecay(prune_rate=args.prune_rate*0.01, T_max=len(self.train_loader)*self.train_epochs)
        # prune initializaiton
        for name, module in self.model.module.named_modules():
            print(name)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if name == 'conv1' or name == 'features.0':
                    if args.prune_conv1:
                        self.parameters_to_prune.append((module, 'weight'))
                    else:
                        print('skip the first conv2d for L1 unstructure global pruning')
                else:
                    self.parameters_to_prune.append((module, 'weight'))
        for name, module in self.model.module.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.initial_num_weights.append(module.weight.numel())
        self.initial_num_weights = np.array(self.initial_num_weights)
        self.parameters_to_prune = tuple(self.parameters_to_prune)
        self.iter_per_epoch = len(self.train_loader)
        self.total_iter = len(self.train_loader) * args.train_epochs
        if args.prune_type == 'local':
            for layer, name in self.parameters_to_prune:
                prune.random_unstructured(layer, name=name, amount=0)
                self.alpha = np.log(args.target_ratio*0.01)
        elif args.prune_type == 'global':
            try: 
                prune.random_unstructured(self.model.module.conv1, 'weight', amount=0.)
            except:
                prune.random_unstructured(self.model.module.features[0], 'weight', amount=0.)
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
                for name, module in self.model.module.named_modules():
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
            for name, module in self.model.module.named_modules():
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
            for name, module in self.model.module.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.weight *= module.weight_mask
        if torch.cuda.is_available():
            if args.gpu:
                self.device = torch.device('cuda:{}'.format(args.gpu))
            else:
                self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        writer = SummaryWriter(self.save_path)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            print(" ! Weight decay NOT applied to BN parameters ")
            parameters = list(self.model.module.named_parameters())
            bn_params = [v for n, v in parameters if 'bn' in n]
            rest_params = [v for n, v in parameters if not 'bn' in n]
            print(len(bn_params))
            print(len(rest_params))
            optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay' : 0},
                                {'params': rest_params, 'weight_decay' : self.args.weight_decay}],
                            self.args.lr,
                            momentum=self.args.momentum,
                            weight_decay=self.args.weight_decay,
                            nesterov = False)
        else:
            raise Exception('wrong optimizer, has to be adam or sgd')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], last_epoch=-1)
        bestacc = 0.0
        best_accuracy = 0
        train_epochs = args.train_epochs
        comp = np.zeros(train_epochs,float)
        bestacc = np.zeros(train_epochs,float)
        testacc = np.zeros(train_epochs, float)
        sparsity_ = np.zeros(train_epochs, float)
        all_loss = np.zeros(train_epochs,float)
        valacc = np.zeros(train_epochs,float)
        early_stop_trigger = 0
        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros_lth(self.model.module.named_modules(), writer, 0)
        sparsity = round(100.0-comp1, 1)
        sparsity_[0] = sparsity
        comp[0] = comp1
        pbar = tqdm(range(args.train_epochs))
        for train_epoch in pbar:
            # Frequency for Testing
            if args.distributed:
                self.train_loader.sampler.set_epoch(train_epoch)
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
            
            lr_scheduler.step()

            all_loss[train_epoch] = loss
            # Frequency for Printing Accuracy and Loss
            if (train_epoch % args.print_freq == 0) and (self.args.val_set == True):
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} Loss: {loss:.6f} Prune rate: {self.prune_rate_decay.get_dr()} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            else:
                pbar.set_description(
                    f'Train Epoch: {train_epoch}/{args.train_epochs} LR: {optimizer.param_groups[-1]["lr"]} Loss: {loss:.6f} Prune rate: {self.prune_rate_decay.get_dr()}')
            if args.early_stop is not None and early_stop_trigger > args.early_stop:
                break
            comp1 = utils.print_nonzeros_lth(self.model.module.named_modules(), writer, train_epoch)
            sparsity_[train_epoch] = round(100.0 - comp1, 1)
            
            if self.args.val_set == False and self.args.fixed_budget == True:
                test_accuracy = self.test(self.model, self.test_loader, criterion)
            if self.args.val_set == False and self.args.fixed_budget == False:
                if (train_epoch % args.valid_freq == 0):
                    test_accuracy = self.test(self.model, self.test_loader, criterion)
                    print(f'Test Accuracy: {test_accuracy}')
                    writer.add_scalar('Accuracy_sparsity/test', test_accuracy, sparsity)
                    writer.add_scalar('Accuracy_epoch/test', test_accuracy, train_epoch)
                    testacc[train_epoch] = test_accuracy
            if self.args.val_set == True and self.args.fixed_budget == True:
                best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                test_accuracy = self.test(best_val_model, self.test_loader, criterion)
            if self.args.val_set == True and self.args.fixed_budget == False:
                if train_epoch > int(np.ceil(self.args.end_update_iter_ratio * self.args.train_epochs)):
                    best_val_model = torch.load(os.path.join(self.save_path, f"best_val_model_{args.prune_type}.pt"))
                    test_accuracy = self.test(best_val_model, self.test_loader, criterion)
                else:
                    test_accuracy = self.test(self.model, self.test_loader, criterion)

            d = {'sparsity': sparsity_[: train_epoch+1], 'testacc': testacc[:train_epoch+1]}
            df = pd.DataFrame(data=d)
            df.to_csv(f"{self.save_path}/sparsity_vs_testacc.csv")
        torch.save(self.model, os.path.join(self.save_path, f"final_model_{args.prune_type}.pt"))



    def train(self, model, train_loader, optimizer, criterion, train_epoch, args):
        metric = MeanMetric()
        model.module.train()
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
            elif self.args.add_criterion == "random" or self.args.add_criterion == "|w|":
                self.importance_scores_add = None
            self.prune_rate_decay.step()
            optimizer.step() 
            if train_iter <= int(self.args.end_update_iter_ratio * self.total_iter):
                self.prune_and_reconnect(train_iter)
            else:
                pass
            print(f"Epoch {train_epoch}, Batch {batch_idx}, Loss {train_loss}")
        return metric.compute().item()

    def test(self, model, test_loader, criterion):
        model.module.eval()
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
                for name, module in self.model.module.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune.remove(module, 'weight')
                prune_ratio = 1. - self.schedule_function(train_iter)
                add_ratio = self.prune_rate_decay.get_dr() * self.schedule_function(train_iter)
                if self.args.prune_criterion == "|w|":
                    global_unstructure.global_unstructured(self.parameters_to_prune, pruning_method=Prune_and_Reconnect, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores=self.importance_scores_prune)
                elif self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_add_rankGrad_Add_Random, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)
                elif self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_mul_rankGrad_Add_Random, amount_prune=prune_ratio+add_ratio, amount_add=add_ratio, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)


            elif self.args.add_criterion == "|grad|":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.module.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                remain = self.initial_num_weights.sum() - already_pruned.sum()
                to_prune_t = to_prune
                to_add_t = int((remain - to_prune_t) * self.prune_rate_decay.get_dr())
                if self.args.prune_criterion == "|w|":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_GradfromW_Add_Grad, amount_prune=(to_prune_t+to_add_t), amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
                elif self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_add_rankGrad_Add_Grad, amount_prune=(to_prune_t+to_add_t), amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)
                elif self.args.prune_criterion == "Rank(|w|) * Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_mul_rankGrad_Add_Grad, amount_prune=(to_prune_t+to_add_t), amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_prune1)
            elif self.args.add_criterion == "":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.module.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                if self.args.prune_criterion == "Rank(|w|) + Rank(|grad|)":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_rankW_add_rankGrad, amount_prune=prune_ratio, amount_add=0, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
            elif self.args.add_criterion == "|w|":
                already_pruned = np.array([int(torch.count_nonzero(module.weight==0)) for name, module in self.model.module.named_modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])
                prune_ratio = 1. - self.schedule_function(train_iter)
                to_prune = int(np.floor((self.initial_num_weights.sum() * prune_ratio - already_pruned.sum())))
                remain = self.initial_num_weights.sum() - already_pruned.sum()
                to_add_t = int(0.5*remain) - to_prune
                to_prune_t = int(0.5*remain)

                if self.args.prune_criterion == "|grad|":
                    global_unstructure_double_importance_scores.global_unstructured_with_different_criteria(self.parameters_to_prune, pruning_method=Prune_WfromGrad_Add_Grad,  amount_prune=to_prune_t, amount_add=to_add_t, importance_scores_prune=self.importance_scores_prune, importance_scores_add=self.importance_scores_add)
            
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
    
def main():
    proposed_prune = Proposed_prune(args)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(proposed_prune.prune, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        proposed_prune.prune(args.gpu, ngpus_per_node, args)
    proposed_prune.prune()
if __name__ == "__main__":
    main()

        

