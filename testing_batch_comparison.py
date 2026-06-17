from algorithms.algo1 import algo1_diffusion_inverse
from algorithms.algo2 import algo2_kwatra_original
from algorithms.algo3 import algo3_kwatra_modifie
from algorithms.algo4 import algo4_nifty

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

from data_tensor_loader import load_data_to_tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### MAIN PARAMETERS
""" 
Availible datasets :
celeba  (c)         cifar   (ci)            cat     (ca)
fashion (f)         mnist   (m)             afhq    (af)
art     (a)         obama   (o)             flickr  (fl)
dog     (d)         grumpy  (g)             panda   (p)
"""


force_reload_tensor = False

dataset_loading_parameters = {
    "data_set_name" : "afhq",
    "num_samples" : 500,
    "target_labels" : [1],
    "image_size" : 28,    
}

# Sub parameters (no specific need to change them by default)
data_dir = "./data"

patch_size = 13
iteration = 20

display_algo_2 = True
image_on_line = 2

torch.cuda.empty_cache()
data_tensor = load_data_to_tensor(None, data_dir=data_dir, dataset_loading_parameters=dataset_loading_parameters, force_reload_tensor=force_reload_tensor)
data_tensor = data_tensor.to(device)


def tensor_to_img(tensor):
    img = np.clip((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)
    # If the last dimension is 1, drop it
    if img.shape[-1] == 1:
        img = img.squeeze(-1) 
        
    return img

## Generation
number_of_images = image_on_line * 4
fig, axes = plt.subplots(4, image_on_line, figsize=(image_on_line, 3))
axes_flat = np.ndarray.flatten(axes)

all_sequences = []
seeds = []

if data_tensor.shape[1] == 1 : plt.set_cmap("gray")
# Algo
from torchvision.utils import save_image
for i in range(number_of_images)[:image_on_line]:
    s = time()
    seeds.append(i+torch.randint(0,99999999,(1,1)))
    r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine',seed=seeds[-1])
    print(f"Execution time: {time() - s:.5f}s")
    last_tensor = r1[-1]
    all_sequences.append(r1)

    plt.imsave(f"ref_img{i}.png",tensor_to_img(last_tensor))

    axes_flat[i].imshow(tensor_to_img(last_tensor))

for i in range(number_of_images)[image_on_line:image_on_line*2]:
    if display_algo_2 :
        s = time()
        seed = seeds[i%image_on_line]
        r1 = algo2_kwatra_original(data_tensor, patchsize=patch_size, N=iteration, device=device,seed=seed)
        
        print(f"Execution time: {time() - s:.5f}s")
        while len(r1) < iteration:
            r1.append(r1[-1])
            
        last_tensor = r1[-1]
        axes_flat[i].imshow(tensor_to_img(last_tensor))
        plt.imsave(f"ref_img{i}.png",tensor_to_img(last_tensor))
        all_sequences.append(r1)
    else : 
        all_sequences.append([torch.ones_like(all_sequences[-1][0]) for i in range(image_on_line)])
    

    
for i in range(number_of_images)[image_on_line*2:image_on_line*3]:
    s = time()
    seed = seeds[i%image_on_line]
    r1 = algo3_kwatra_modifie(data_tensor, patchsize=patch_size, N=iteration, device=device,seed=seed)
    while len(r1) < iteration:
            r1.append(r1[-1])
    print(f"Execution time: {time() - s:.5f}s")
    
    last_tensor = r1[-1]
    all_sequences.append(r1)
    plt.imsave(f"ref_img{i}.png",tensor_to_img(last_tensor))
    axes_flat[i].imshow(tensor_to_img(last_tensor))
    

for i in range(number_of_images)[image_on_line*3:image_on_line*4]:
    s = time()
    seed = seeds[i%image_on_line]
    r1 = algo4_nifty(data_tensor, patchsize=patch_size, N=iteration, device=device,seed=seed)
    
    print(f"Execution time: {time() - s:.5f}s")
    plt.imsave(f"ref_img{i}.png",tensor_to_img(last_tensor))
    last_tensor = r1[-1]
    all_sequences.append(r1)
    print(len(r1))
    axes_flat[i].imshow(tensor_to_img(last_tensor))
    
    
    
[ax.axis('off') for ax in axes_flat]

ims = []
num_steps = min(len(seq) for seq in all_sequences) # because sometimes algo 2 and 3 stops sooner (loop break if there's no change, line 7)
for step in range(num_steps):
    frame_artists = []
    for i in range(number_of_images):
        img_data = tensor_to_img(all_sequences[i][step])
        im = axes_flat[i].imshow(img_data, animated=True)
        frame_artists.append(im)
    
    ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=2000)
plt.show()