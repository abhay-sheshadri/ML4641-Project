import pickle
import torch
from PIL import Image
import numpy as np


def load_stylegan_generator(file_path, requires_grad=False):
    with open(file_path, 'rb') as f:
        G_temp = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
        return G_temp


G = load_stylegan_generator("anime.pkl")
G.eval()


def generate_random_w(n=1):
    z = torch.randn([n, G.z_dim]).cuda()
    w = G.mapping(z, None, truncation_psi=0.7, truncation_cutoff=8)
    return w


def convert_torch_to_pil(image_tensor):
    img = (image_tensor.permute(0, 2, 3, 1) *
           127.5 + 128).clamp(0, 255).to(torch.uint8)
    images = []
    for i in range(img.shape[0]):
        if img.shape[-1] == 3:
            images.append(Image.fromarray(img[i].cpu().numpy(), 'RGB'))
        elif img.shape[-1] == 4:
            images.append(Image.fromarray(img[i].cpu().numpy(), 'RGBA'))
    return images


def generate_image(w=None, coords=[0, 0], scale=1, as_pil=True):
    if w is None:
        w = generate_random_w()
    if coords is not None:
        m = np.eye(3)
        m[0][2] = coords[0]
        m[1][2] = coords[1]
        m[0][0] = scale
        m[1][1] = scale
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    img = G.synthesis(w, noise_mode="const",)
    if as_pil:
        return convert_torch_to_pil(img)
    else:
        return img
