from algorithms.algo1 import algo1_diffusion_inverse
from algorithms.algo2 import algo2_kwatra_original
from algorithms.algo3 import algo3_kwatra_modifie

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from PIL import Image
import torchvision.transforms as transforms

import dataset_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF = True

def tensor_to_img(tensor):
    return np.clip((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)

# Parameters
## Training data
used_data_set = 2
match used_data_set:
    case 0: data_tensor = dataset_loader.load_celeba(root="./data", num_samples=200, target_attr_indices=[4], match_any=False, image_size=32)
    case 1: data_tensor = dataset_loader.load_cifar10(root="./data", num_samples=200, target_labels=[3])
    case 2: data_tensor = dataset_loader.load_mnist(root="./data", num_samples=500, target_labels=[3,5,4])
    case 3: data_tensor = dataset_loader.load_mnist_fashion(root="./data", num_samples=500, target_labels=[0,8,9])

## Generation
patch_size = 9
iteration = 10

image_count = 60
image_on_line = 5

image_on_row = image_count // image_on_line

fig, axes = plt.subplots(image_on_row + 1, image_on_line, figsize=(image_on_line, image_on_row + (1 if image_on_row == 0 else 0)))
axes_flat = np.ndarray.flatten(axes)

all_sequences = []

if data_tensor.shape[1] == 1 : plt.set_cmap("gray")

# Algo
for i in range(image_count):
    s = time()
    #r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine')
    #r1 = algo2_kwatra_original(data_tensor, patchsize=patch_size, N=iteration, device=device)
    r1 = algo3_kwatra_modifie(data_tensor, patchsize=patch_size, N=iteration, device=device)
    print(f"Execution time: {time() - s:.5f}s")
    
    last_tensor = r1[-1]
    
    all_sequences.append(r1)
    
    axes_flat[i].imshow(tensor_to_img(last_tensor))

[ax.axis('off') for ax in axes_flat]

ims = []
num_steps = min(len(seq) for seq in all_sequences) # des fois avec les algos 2 ou 3, ça peut s'arreter avt

for step in range(num_steps):
    frame_artists = []
    for i in range(image_count):
        img_data = tensor_to_img(all_sequences[i][step])
        im = axes_flat[i].imshow(img_data, animated=True)
        frame_artists.append(im)
    
    ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=5000)
plt.show()