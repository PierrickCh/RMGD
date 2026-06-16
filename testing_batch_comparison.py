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
celeba  (c)
cifar   (ci)
fashion (f)
art     (a)
cat     (ca)
dog     (d)
grumpy  (g)
obama   (o)
panda   (p)
flickr  (fl)
mnist   (m)
"""
data_set_name = "c"
subset_name = 30 # Used to specify a specific appendix needed for the file load/saving (ex if you use 50s for the sample number, if you don't check the force reload but there are no 50s file, it'll create one, and if there's one, it'll load it)

force_reload_tensor = False

dataset_loading_parameters = {
    "num_samples" : 250,
    "target_labels" : [],
    "image_size" : 32,    
}

# Sub parameters (no specific need to change them by default)
data_dir = "./data"

patch_size = 9
iteration = 20

display_algo_2 = True
image_on_line = 2

torch.cuda.empty_cache()
data_tensor = load_data_to_tensor(None, data_dir=data_dir, data_set_name=data_set_name, subset_name=subset_name, dataset_loading_parameters=dataset_loading_parameters, force_reload_tensor=force_reload_tensor)
data_tensor = data_tensor.to(device)


def tensor_to_img(tensor):
    return np.clip((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)

## Generation
number_of_images = image_on_line * 4
fig, axes = plt.subplots(4, image_on_line, figsize=(image_on_line, 3))
axes_flat = np.ndarray.flatten(axes)

all_sequences = []
seeds = []
if data_tensor.shape[1] == 1 : plt.set_cmap("gray")
# Algo
for i in range(number_of_images)[:image_on_line]:
    s = time()
    seeds.append(i+torch.randint(0,99999999,(1,1)))
    r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine',seed=seeds[-1])
    print(f"Execution time: {time() - s:.5f}s")
    last_tensor = r1[-1]
    all_sequences.append(r1)
    
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
    
    axes_flat[i].imshow(tensor_to_img(last_tensor))
    
    
for i in range(number_of_images)[image_on_line*3:image_on_line*4]:
    s = time()
    seed = seeds[i%image_on_line]
    r1 = algo4_nifty(data_tensor, patchsize=patch_size, N=iteration, device=device,seed=seed)
    
    print(f"Execution time: {time() - s:.5f}s")
    
    last_tensor = r1[-1]
    all_sequences.append(r1)
    
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

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=5000)
plt.show()