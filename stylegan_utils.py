import pickle
import torch
from PIL import Image
import numpy as np
from networks_stylegan3 import Generator
from sklearn.cluster import KMeans


def load_stylegan_generator(file_path, return_raw=False, requires_grad=False):
    with open(file_path, 'rb') as f:
        G_temp = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
        if return_raw:
            return G_temp
    G = Generator(
        G_temp.z_dim,
        G_temp.c_dim,
        G_temp.w_dim,
        G_temp.img_resolution,
        G_temp.img_channels,
        channel_base=16384).cuda()
    G.load_state_dict(G_temp.state_dict())
    for p in G.parameters():
        p.requires_grad = requires_grad
    return G


G = load_stylegan_generator("anime.pkl", return_raw=True)
for p in G.parameters():
    p.requires_grad = True
G.eval()

def generate_random_w(n=1):
    z = torch.randn([n, G.z_dim]).cuda()
    w = G.mapping(z, None, truncation_psi= 0.7, truncation_cutoff=8)
    return w

def convert_torch_to_pil(image_tensor):
    img = (image_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
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
    img, maps = G.synthesis(w, noise_mode="const", return_maps=True)
    for i in range(13):
        maps[i] = maps[i][:, :, 10:-10, 10:-10].to(torch.float32)
    maps[-2] = maps[-2].to(torch.float32)
    if as_pil:
        return convert_torch_to_pil(img), maps[:-1]
    else:
        return img, maps[:-1]


def clustering_segmentation_maps(map_layer=11):

    X = []
    for i in range(25):
        im, maps = generate_image()
        X.append(
            maps[map_layer].cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        )
        
    X = np.stack(X)
    X = X.reshape(-1, X.shape[-1])
    print("Processing", X.shape[0], "vectors...")
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

    colors = [tuple(np.random.choice(range(256), size=3)) for i in range(10)]

    for k in range(5):
        img1, maps = generate_image()
        img2 = Image.new("RGB", (maps[map_layer].shape[2], maps[map_layer].shape[3]))
        pixels = img2.load()
        for i in range(maps[map_layer].shape[2]):
            for j in range(maps[map_layer].shape[3]):
                vec = maps[map_layer][:, :, i, j].cpu().detach().numpy()
                seg = kmeans.predict(vec)[0]
                pixels[j, i] = colors[seg]
        
        img1.save(f"raw_{k}.png")
        img2.save(f"seg_{k}.png")

