from stylegan_utils import *
from resnet import ResnetModel
from datasets import AnimeDataset

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import os

hyperparams = {
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "resize": (32, 32),

    "use_synthetic": True,
    "num_batches": 10,

    "num_test": 3
}


def train_with_synthetic():

    # Initialize model
    net = ResnetModel(1, 512, n_layers=50, res=hyperparams["resize"]).cuda()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=hyperparams["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hyperparams["num_epochs"])

    for epoch in range(hyperparams["num_epochs"]):

        net.train()
        for batch_idx in range(hyperparams["num_batches"]):

            optimizer.zero_grad()

            # Sample images from Gan distribution
            with torch.no_grad():
                w = generate_random_w(16)
                img = G.synthesis(w, noise_mode="const",)

            # Downsample
            img = F.interpolate(
                img, hyperparams["resize"]).mean(1, keepdim=True)
            pred_latent = net(img)
            loss = F.mse_loss(pred_latent, w[:, 0, :])

            # improve
            loss.backward()
            optimizer.step()

        net.eval()
        test_distances = []
        with torch.no_grad():
            for i in range(hyperparams["num_test"]):
                # Generate an initial image to put through the network
                w = generate_random_w(1)
                img = G.synthesis(w, noise_mode="const",)
                imgd = F.interpolate(
                    img, hyperparams["resize"]).mean(1, keepdim=True)
                pred_latent = net(imgd).unsqueeze(0).repeat(1, 16, 1)
                imgd = F.interpolate(imgd, (256, 256)).repeat(1, 3, 1, 1)
                # Add tests
                images = torch.cat([img, imgd], dim=3)
                for j in range(3):
                    w2 = generate_random_w(1)
                    pred_latent[0, 8:, :] = w2[0, 8:, :]
                    imgr = G.synthesis(pred_latent, noise_mode="const",)
                    images = torch.cat([images, imgr], dim=3)

                images = images.detach().cpu()
                convert_torch_to_pil(images)[0].save(f"test_{i}.jpg")
                test_distances.append(F.mse_loss(
                    w[0, 0, :], pred_latent[0, 0, :]).item())

        print(
            f"Epoch: {epoch+1}, MSE Latent Loss: {sum(test_distances) / len(test_distances)}")
        torch.save(net.state_dict(), "weights.pkl")
        scheduler.step()


def train_with_real():

    # Initialize model
    net = ResnetModel(3, 512, res=hyperparams["resize"])
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
            rs_images = F.interpolate(gt_images, hyperparams["resize"])


if __name__ == "__main__":
    if hyperparams["use_synthetic"]:
        train_with_synthetic()
    else:
        train_with_real()
