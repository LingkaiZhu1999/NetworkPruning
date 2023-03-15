#ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import io
import utils

#ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model, writer, _ite):
    nonzero = total = 0
    info_ = ''
    if list(model.named_buffers()) != []:
        model_info = model.named_buffers()
    else:
        model_info = model.named_parameters()   
    for name, p in model_info:
        tensor = p.data
        nz_count = torch.count_nonzero(tensor).cpu().numpy()
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        info = f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}'
        print(info)
        info_ += info
    writer.add_text("prune info", info_, _ite)
    summary_info = f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned) \n'
    print(summary_info)
    writer.add_text("prune summary", summary_info, _ite)
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

def plot_sparsity_testacc(sparsity, testacc, plot_path):
    fig = plt.figure()
    plt.plot(sparsity, testacc)
    plt.xlabel('Sparsity')
    # plt.xticks(sparsity)
    # plt.set_
    plt.ylabel('Test accuracy')
    # plt.ylim([70,100])
    plt.savefig(os.path.join(plot_path, "acc_vs_sparsity.png"), dpi=1200)
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