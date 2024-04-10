#ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import io
import utils
import torch.nn as nn

def print_nonzeros(model_info, save_path, _ite):
    nonzero = total = 0
    info_ = '' 
    for name, p in model_info:
        if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
            weight = p.weight.data
            bias = p.bias.data
            nz_count_weight = torch.count_nonzero(weight).cpu().numpy()
            total_params_weight = np.prod(weight.shape)
            nz_count_bias = torch.count_nonzero(bias).cpu().numpy()
            total_params_bias = np.prod(bias.shape)
            nonzero += (nz_count_weight + nz_count_bias)
            total += (total_params_weight + total_params_bias)
            info = f'{name+".weight":20} | nonzeros = {nz_count_weight:7} / {total_params_weight:7} ({100 * nz_count_weight / total_params_weight:6.2f}%) | total_pruned = {total_params_weight - nz_count_weight :7} | shape = {weight.shape}\n'
            info_ += info
            info = f'{name+".bias":20} | nonzeros = {nz_count_bias:7} / {total_params_bias:7} ({100 * nz_count_bias / total_params_bias:6.2f}%) | total_pruned = {total_params_bias - nz_count_bias :7} | shape = {bias.shape}\n'
            info_ += info
    info_ += f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned) \n'
    print(info_)
    info_ += '--------------------------------------------------------------------------------------------\n'
    # writer after the previous without the first line empty

    with open(os.path.join(save_path, f"prune_summary.txt"), 'a') as f:
        f.write(info_)
    return (round((nonzero/total)*100,1))

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = (mask_temp[step] * initial_state_dict[name]).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

def plot_sparsity_testacc(sparsity, testacc, plot_path, name='test'):
    fig = plt.figure()
    plt.plot(sparsity, testacc, 'o-')
    plt.xlabel('Sparsity')
    # plt.xticks(sparsity)
    # plt.set_
    plt.ylabel('Test accuracy')
    # plt.ylim([70,100])
    plt.savefig(os.path.join(plot_path, f"acc_vs_sparsity_{name}.png"), dpi=1200)
    plt.close()
    return fig




#ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

#FIXME 
def plot_train_test_stats(stats,
                          epoch_num,
                          key1='train',
                          key2='test',
                          key1_label=None,
                          key2_label=None,
                          xlabel=None,
                          ylabel=None,
                          title=None,
                          yscale=None,
                          ylim_bottom=None,
                          ylim_top=None,
                          savefig=None,
                          sns_style='darkgrid'
                          ):

    assert len(stats[key1]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key1, len(stats[key1]), epoch_num)
    assert len(stats[key2]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key2, len(stats[key2]), epoch_num)

    plt.clf()
    sns.set_style(sns_style)
    x_ticks = np.arange(epoch_num)

    plt.plot(x_ticks, stats[key1], label=key1_label)
    plt.plot(x_ticks, stats[key2], label=key2_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fancybox=True)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()