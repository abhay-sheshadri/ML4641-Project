import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import glob
from PIL import Image


class AnimeDataset(torch.utils.data.Dataset):

    def __init__(self, image_folder=r"C:\Users\abhay\Documents\datasets\danbooru_edit"):

        self.image_paths = list(glob.glob(image_folder+"\*.jpg"))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        im = Image.open(image_path)
        image = torchvision.transforms.functional.pil_to_tensor(im)
        image = image.float() / 255
        image = (image * 2) - 1
        return image

    def __len__(self,):
        return len(self.image_paths)
