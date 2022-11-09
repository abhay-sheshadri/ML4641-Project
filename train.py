from stylegan_utils import *
from resnet import ResnetModel
from datasets import AnimeDataset

import torch
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import os

hyperparams = {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,

    "save_interval": 4096,
    "lpips_lambda": 0.8,
    "l2_lambda": 1,
    "w_norm_lambda": 0,
    "moco_lambda": 0.5,

    "input_nc": 6,
    "n_iters_per_batch": 5,
    "output_size": 256,
    "n_styles": 2,
    "n_layers": 18,

    "load_prev": False
}


def main():

    # Initialize model
    net = ResnetModel(3, 512)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=hyperparams["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hyperparams["num_epochs"])

    # Initialize dataset
    dataset = AnimeDataset()
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.shuffle(indices)

    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(valid_indices)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler,
        batch_size=hyperparams["batch_size"], )
    test_loader = torch.utils.data.DataLoader(
        dataset, sampler=val_sampler,
        batch_size=hyperparams["batch_size"],)

    # Training loop
    
    for epoch in range(hyperparams["num_epochs"]):

        for idx, gt_images in enumerate(train_loader):
            gt_images = gt_images.cuda()


if __name__ == "__main__":
    main()
