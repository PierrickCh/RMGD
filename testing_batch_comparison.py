from algorithms.algo1 import algo1_diffusion_inverse
from algorithms.algo2 import algo2_kwatra_original
from algorithms.algo3 import algo3_kwatra_modifie
from algorithms.algo4 import algo4_nifty

import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from PIL import Image
import torchvision.transforms as transforms

import dataset_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### MAIN PARAMETERS
"""
celeba
cifar
fashion
art
cat
dog
grumpy
obama
panda
flickr
mnist
"""


used_data_set = 6
force_reload_tensor = False

patch_size = 9
iteration = 20

display_algo_2 = True

image_on_line = 2



torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF = True

def tensor_to_img(tensor):
    return np.clip((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2, 0, 1)

# loading/saving of cahce
cache_dir = "data"
os.makedirs(cache_dir, exist_ok=True)

status_file = os.path.join(cache_dir, "last_run_data.txt")
tensor_file = os.path.join(cache_dir, f"tensor_cache_{used_data_set}.pt")

load_from_cache = False
if not force_reload_tensor and os.path.exists(status_file) and os.path.exists(status_file):
    with open(status_file, 'r') as f:
        last_set = f.read().strip()
        if last_set == str(used_data_set) or os.path.exists(tensor_file):
            load_from_cache = True

if load_from_cache:
    print(f"Loading tensor from : {tensor_file}")
    data_tensor = torch.load(tensor_file, map_location=device)
else:
    print(f"Processing dataset {used_data_set} and creating tensor file")
    match used_data_set:
        case 0: data_tensor = dataset_loader.load_celeba(root="./data", num_samples=100, target_attr_indices=[], match_any=False, image_size=64)
        case 1: data_tensor = dataset_loader.load_cifar10(root="./data", num_samples=500, target_labels=[3])
        case 2: data_tensor = dataset_loader.load_mnist(root="./data", num_samples=1500, target_labels=[5,4,3])
        case 3: data_tensor = dataset_loader.load_mnist_fashion(root="./data", num_samples=500, target_labels=[0,8,9])
        case 4: data_tensor = dataset_loader.load_jpg_folder(folder_path="./data/flickr8k/Images",num_samples=100,image_size=32)
        case 5 : data_tensor = dataset_loader.load_few_shot_panda(image_size=64)
        case 6 : data_tensor = dataset_loader.load_parquet(file_path_or_url="data/few-shot-grumpy-cat/train-00000-of-00001.parquet",image_size=64)
    # Save the tensor and update the file
    torch.save(data_tensor, tensor_file)
    with open(status_file, 'w') as f:
        f.write(f"{used_data_set}")

data_tensor = data_tensor.to(device)

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