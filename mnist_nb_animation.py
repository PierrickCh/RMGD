from algorithms.algo1 import algo1_diffusion_inverse
import struct
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_mnist_idx3(file_path, num_samples=None):
    """
    Loads MNIST images from an idx3-ubyte file.
    Returns a tensor of shape (N, 1, H, W) normalized to [-1, 1].
    N : number of image
    1 : color channel
    H : number of pixels (height)
    W : number of pixels (width)
    """
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16)) # https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
    
    if num_samples is not None:
        images = images[:num_samples] # Take the first num_samples
        
    img_tensor = torch.from_numpy(images).float() / 255.0
    img_tensor = img_tensor * 2 - 1 # Scale to [-1, 1]
    return img_tensor.to(device)

# Parameters
## Training data
file_path = 'data/train-images.idx3-ubyte'
data_tensor = load_mnist_idx3(file_path, 1500)

## Display
batch_size_sq = 3

## Generation
patch_size = 9 # lke 3x3, 9x9
iteration = 5
noise_scheduling_method = "cosine" # "linear"


fig, axes = plt.subplots(batch_size_sq, batch_size_sq, figsize=(batch_size_sq,batch_size_sq))

axes = axes.flatten()

# Collect all denoising sequences
all_sequences = []
for i in range(batch_size_sq**2):
    r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine')
    all_sequences.append(r1)
    im = axes[i].imshow(r1[-1][0,0].cpu(), cmap='gray', animated=True)
    axes[i].axis('off')

# Create animation frames
ims = []
max_steps = max(len(seq) for seq in all_sequences)

for step in range(max_steps):
    frame_artists = []
    for i in range(batch_size_sq**2):
        current_step = min(step, len(all_sequences[i]) - 1)
        img_data = (all_sequences[i][current_step][0, 0].cpu() + 1) / 2
        
        im = axes[i].imshow(img_data, cmap='gray', animated=True)
        axes[i].axis('off')
        frame_artists.append(im)
    
    ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=5000)
plt.tight_layout()
plt.show()