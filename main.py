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
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle
import torch.nn.utils.prune as prune
# Custom Libraries
import utils
import PIL
# Plotting Style
sns.set_style('darkgrid')

parser = argparse.ArgumentParser()
parser.add_argument("--lr",default=0.0002, type=float, help="Learning rate") # learning rate have a big effect
parser.add_argument("--batch_size", default=60, type=int)
parser.add_argument("--start_prune_prune_round", default=0, type=int)
parser.add_argument("--train_epochs", default=500, type=int)
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--early_stop", default=15, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--retrain_type", default="original", type=str, help="original | reinit")
parser.add_argument("--prune_type", default="local", help="local | global")
parser.add_argument("--gpu", default="1", type=str)
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="lenet5", type=str, help="fc1 | advanced_dropout_fc | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
parser.add_argument("--prune_rounds", default=30, type=int, help="Pruning args.prune_roundss count")
parser.add_argument("--weight_decay", default=1.2e-3, type=float, help="weight decay for adam optim")
parser.add_argument("--seed", default=1, type=int)


args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
with open(f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/args.txt", 'w') as f:
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)), file=f) 
# Main
def main(args, ITE=0):
    # tensorboard
    writer = SummaryWriter(f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.retrain_type=="reinit" else False
    torch.cuda.manual_seed_all(args.seed)

    mean = {
        'mnist': (0.1307,),
        'cifar10': (0.4914, 0.4822 ,0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408)
    }
    std = {
        'mnist': (0.3081,),
        'cifar10': (0.2470, 0.2435, 0.2616),
        'cifar100': (0.2675, 0.2565, 0.2761),
    }
    # Data Loader
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean[args.dataset], std[args.dataset])])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, advanced_dropout_fc, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet  
    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_dataset, val_dataset = torch.utils.data.random_split(traindataset, [55000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
    #train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=True)
    
    # Importing Network Architecture
    global model
    if args.arch_type == "fc1":
        model = fc1.fc1().to(device)
    elif args.arch_type == "advanced_dropout_fc":
        model = advanced_dropout_fc.advanced_drop_fc().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    # If you want to add extra model paste here
    else:
        print("\nWrong Model choice\n")
        exit()
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    # Weight Initialization
    model.apply(weight_init)


    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/initial_state_dict_{args.retrain_type}.pt")
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning args.prune_rounds is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    comp = np.zeros(args.prune_rounds,float)
    bestacc = np.zeros(args.prune_rounds,float)
    testacc = np.zeros(args.prune_rounds, float)
    sparsity_ = np.zeros(args.prune_rounds, float)
    step = 0
    all_loss = np.zeros(args.train_epochs,float)
    all_accuracy = np.zeros(args.train_epochs,float)
    early_stop_trigger = 0

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # if name == 'conv1':
            #     if args.prune_conv1:
            #         parameters_to_prune.append((module, 'weight'))
            #     else:
            #         print('skip the first conv2d for L1 unstructure global pruning')
            # else:
            parameters_to_prune.append((module, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
        
    for prune_round in range(args.start_prune_prune_round, args.prune_rounds):
        if not prune_round == 0: # don't prune for the first running prune_round, because we want the model to be well trained before we prune it.
            print(prune_round)
            # prune_percent_each_prune_round = np.power(args.prune_percent, 1/args.prune_rounds) * 0.01 # p^(1/n) %
            # prune_percent_each_prune_round = args.prune_percent * 0.01 / args.prune_rounds 
            prune_by_percentile(args.prune_percent * 0.01, parameters_to_prune)
            mask = get_mask(mask, model)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = (param.data * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\n--- Pruning Level [{ITE}:{prune_round}/{args.prune_rounds}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model, writer, prune_round)
        sparsity = round(float(100.0 - comp1), 1)
        sparsity_[prune_round] = sparsity
        comp[prune_round] = comp1
        pbar = tqdm(range(args.train_epochs))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                val_accuracy = test(model, val_loader, criterion)
                writer.add_scalar(f'{prune_round}/valacc', val_accuracy, iter_)
                # Save Weights
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
                    torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/{prune_round}_model_{args.retrain_type}.pt")
                    early_stop_trigger = 0
                else:
                    early_stop_trigger += 1

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = val_accuracy
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.train_epochs} Loss: {loss:.6f} Val Accuracy: {val_accuracy:.2f}% Best Val Accuracy: {best_accuracy:.2f}%')       
            if early_stop_trigger > args.early_stop:
                break
        best_val_model = torch.load(f"{os.getcwd()}/saves/{args.arch_type}_lr_{args.lr}/{args.dataset}/{prune_round}_model_{args.retrain_type}.pt")
        test_accuracy = test(best_val_model, test_loader, criterion)
        print(f'Test Accuracy: {test_accuracy}')
        writer.add_scalar('Accuracy/val', best_accuracy, sparsity)
        writer.add_scalar('Accuracy/test', test_accuracy, sparsity)
        bestacc[prune_round] = best_accuracy
        testacc[prune_round] = test_accuracy
        fig = utils.plot_sparsity_testacc(sparsity_[:prune_round+1], testacc[:prune_round+1], args)
        writer.add_figure('sparsity_testacc', fig, prune_round)
        # Plotting Loss (Training), Accuracy (Testing), args.prune_rounds Curve
        #NOTE Loss is computed for every args.prune_rounds while Accuracy is computed only for every {args.valid_freq} args.prune_roundss. Therefore Accuracy saved is constant during the uncomputed args.prune_roundss.
        #NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1,(args.train_epochs)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
        plt.plot(np.arange(1,(args.train_epochs)+1), all_accuracy, c="red", label="Accuracy") 
        plt.title(f"Loss Vs Accuracy Vs args.prune_roundss ({args.dataset},{args.arch_type}_lr_{args.lr})") 
        plt.xlabel("args.prune_roundss") 
        plt.ylabel("Loss and Accuracy") 
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_all_accuracy_{comp1}.dat")
        
        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)
        
        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.train_epochs,float)
        all_accuracy = np.zeros(args.train_epochs,float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_rounds)
    plt.plot(a, bestacc, c="blue", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type}_lr_{args.lr})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("test accuracy") 
    plt.xticks(a, comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}_lr_{args.lr}/{args.dataset}/{args.retrain_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()                    
   
# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                grad_tensor = p.grad.data
                grad_tensor = torch.where(torch.abs(tensor) < EPS, 0, grad_tensor)
                p.grad.data = grad_tensor.to(device)

        optimizer.step()
    return train_loss.item()

# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(prune_percent, parameters_to_prune):
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
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percent)

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            # param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            param.data = (mask_temp[step] * initial_state_dict[name[:-5]]).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def get_mask(mask, model):
    step = 0
    for module in model.children():
        mask[step] = list(module.named_buffers())[0][1].cuda()
        step += 1 
    return mask


if __name__=="__main__":
    
    # from gooey import Gooey
    # @Gooey      
    
    # Arguement Parser

    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
