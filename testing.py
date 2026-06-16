from algorithms.algo1 import algo1_diffusion_inverse
from algorithms.algo2 import algo2_kwatra_original
from algorithms.algo3 import algo3_kwatra_modifie
from algorithms.algo4 import algo4_nifty
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Added libraries for image handling
from PIL import Image
import torchvision.transforms as transforms

import dataset_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF=True

def tensor_to_img(tensor):
    return np.clip( (tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)
# Parameters
## Training data
used_data_set = 0
match used_data_set :
    case 0 : data_tensor = dataset_loader.load_celeba(root="./data",num_samples=300, target_attr_indices=[4],match_any=False,image_size=64) # 4 = chauves
    case 1 : data_tensor = dataset_loader.load_cifar10(root="./data",num_samples=200, target_labels=[3]) # 3 = chats
    case 2 : data_tensor = dataset_loader.load_mnist(root="./data",num_samples=500,target_labels=[3,5,4]) # Prendre des 5,4 et 3
    case 3 : data_tensor = dataset_loader.load_mnist_fashion(root="./data",num_samples=500,target_labels=[0,8,9]) # de type chaussure (botte, sandale, et jsp quoi)

## Generation
patch_size = 9 # lke 3x3, 9x9
iteration = 10

image_count = 10
image_on_line = 5

fig, axes = plt.subplots(1,1, figsize=(4,5))
sequence = []

from time import time

#torch.manual_seed(1)
s = time()
#r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine')
#r1 = algo2_kwatra_original(data_tensor, patchsize=patch_size, N=iteration, device=device)
r1 = algo4_nifty(data_tensor, patchsize=patch_size, N=iteration, device=device)

print(time()-s)
plt.imshow(tensor_to_img(r1[-1]), animated=True)
plt.axis('off')

# Create animation frames
ims = []
for step in r1:
    frame_artists = []
    img_data = tensor_to_img(step)
    
    im = plt.imshow(img_data, animated=True)
    plt.axis('off')
    frame_artists.append(im)
    
    ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=5000)
plt.tight_layout()
plt.show()

