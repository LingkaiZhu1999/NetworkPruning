# FGGP: Fixed-Rate Gradient-First Gradual Pruning

You can run the experiments on CIFAR-10 using `main_proposed_class_schedule_cifar.py`.

Code for the paper FGGP: Fixed-Rate Gradient-First Gradual Pruning, if you find this repo helpful, please consider cite our work.
```
@inproceedings{10.1007/978-3-031-95911-0_1,
author = {Zhu, Lingkai and Bezek, Can Deniz and Goksel, Orcun},
title = {FGGP: Fixed-Rate Gradient-First Gradual Pruning},
year = {2025},
isbn = {978-3-031-95910-3},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-95911-0_1},
doi = {10.1007/978-3-031-95911-0_1},
abstract = {In recent years, the increasing size of deep learning models and their growing demand for computational resources have drawn significant attention to the practice of pruning neural networks, while aiming to preserve their accuracy. In unstructured gradual pruning, which sparsifies a network by gradually removing individual network parameters until a targeted network sparsity is reached, recent works show that both gradient and weight magnitudes should be considered. In this work, we show that such mechanism, e.g., the order of prioritization and selection criteria, is essential. We introduce a gradient-first magnitude-next strategy for choosing the parameters to prune, and show that a fixed-rate subselection criterion between these steps works better, in contrast to the annealing approach in the literature. We validate this on CIFAR-10 dataset, with multiple randomized initializations on both VGG-19 and ResNet-50 network backbones, for pruning targets of 90, 95, and 98\% sparsity and for both initially dense and 50\% sparse networks. Our proposed fixed-rate gradient-first gradual pruning (FGGP) approach outperforms its state-of-the-art alternatives in most of the above experimental settings, even occasionally surpassing the upperbound of corresponding dense network results, and having the highest ranking across the considered experimental settings.},
booktitle = {Image Analysis: 23rd Scandinavian Conference, SCIA 2025, Reykjavik, Iceland, June 23–25, 2025, Proceedings, Part I},
pages = {3–15},
numpages = {13},
keywords = {Model compression, image classification},
location = {Reykjavik, Iceland}
}
```