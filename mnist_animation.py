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
file_path = 'data/numbers.idx3-ubyte'
data_tensor = load_mnist_idx3(file_path, 1500)

## Generation
patch_size = 9 # lke 3x3, 9x9
iteration = 5
noise_scheduling_method = "cosine" # "linear"

fig, axes = plt.subplots(1,1, figsize=(10,10))

sequence = []
r1 = algo1_diffusion_inverse(data_tensor, patchsize=patch_size, T=iteration, device=device, schedule='cosine')
plt.imshow(r1[-1][0,0].cpu(), cmap='gray', animated=True)
plt.axis('off')
# Create animation frames
ims = []
for step in r1:
    frame_artists = []
    img_data = (step[0,0].cpu() + 1) / 2
    
    im = plt.imshow(img_data, cmap='gray', animated=True)
    plt.axis('off')
    frame_artists.append(im)
    
    ims.append(frame_artists)

ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=5000)
plt.tight_layout()
plt.show()