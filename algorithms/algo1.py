import torch
import torch.nn.functional as F
from tqdm import tqdm

def extract_centered_patches(img, patchsize):
    """
    Extrait les patches centrés pour chaque pixel de l'image.
    """
    pad = patchsize // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    return F.unfold(img_padded, kernel_size=patchsize, padding=0, stride=1)

def get_noise_schedule(T, schedule, device):
    """
    Noise Schedule de différentes manières
    """
    
    # Les paramètres (ɑt)t∈[T] sont choisis de telle sorte que (ɑ̄)T ≃ 0
    # (ɑ̄)T décroissant, https://arxiv.org/pdf/2412.20292 (2 p3)
    if schedule == 'cosine': 
        # Le schedule Cosine d'après https://arxiv.org/pdf/2102.09672 et https://arxiv.org/h    tml/2502.04669v1
        steps = torch.linspace(0, T, T + 1, device=device)
        s = 0.008
        f_t = torch.cos(((steps / T) + s) / (1 + s) * (torch.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]
        alpha_bar = alpha_bar[1:]
        
        alpha = torch.zeros(T, device=device)
        alpha[0] = alpha_bar[0]
        for t in range(1, T):
            alpha[t] = alpha_bar[t] / alpha_bar[t-1]
        beta = 1.0 - alpha
        beta = torch.clamp(beta, max=0.999)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
    else:
        # Schedule linéaire classique
        beta = torch.linspace(1e-4, 2e-2, T, device=device) # https://arxiv.org/pdf/2006.11239 p11
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
    return alpha, alpha_bar, beta

@torch.no_grad()
def algo1_diffusion_inverse(D_train, patchsize=3, N=1000, device='cpu', schedule='cosine',seed=None):
    """
    Algorithme 1 : Diffusion inverse avec score localement contraint.
    N = T (dans le papier) = nombre d'itérations
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    D_train = D_train.to(device)
    N_imgs, C, H, W = D_train.shape
    # Définition du Noise Schedule
    alpha, alpha_bar, beta = get_noise_schedule(N, schedule, device)
    
    Z = extract_centered_patches(D_train, patchsize) 
    center_p = (patchsize**2) // 2
    c_indices = [c * (patchsize**2) + center_p for c in range(C)]
    Z_center = Z[:, c_indices, :] # Extraction du pixel central de chaque patch d'entraînement
    
    # Echantillonnage initial
    y_t = torch.randn(1, C, H, W, device=device) / torch.sqrt(alpha_bar[-1])
    
    saved_steps = []
    saved_steps.append(y_t.cpu()) # Sauvegarde des images sur le CPU (pour essayer de garder de la VRAM au  plus possible)
    
    for t_step in tqdm(range(N - 1, -1, -1)):
        # Pas de boucle pour 3 car calcul vectoriel
        
        a_t = alpha[t_step]
        a_bar_t = alpha_bar[t_step]
        a_bar_t_prev = alpha_bar[t_step-1] if t_step > 0 else torch.tensor(1.0, device=device)
        
        tau_t = (1.0 - a_bar_t) / a_bar_t # p3 en haut
        lambda_t = (1.0 - a_t) / (1.0 - a_bar_t) # 2.3 - 10

        # σ en fonction du modèle DDPM https://arxiv.org/pdf/2006.11239 (p3 3.2)
        if t_step > 0:
            # Standard DDPM :
            #sigma_t_sq = beta[t_step]
            
            # Alternative DDPM :
            sigma_t_sq = ((1.0 - a_bar_t_prev) / (1.0 - a_bar_t)) * beta[t_step]
            
            sigma_t_p = torch.sqrt(sigma_t_sq / a_bar_t_prev) 
            eps_i = torch.randn_like(y_t)
        else:
            # Dernier pas (t=0) : pas de bruit ajouté
            sigma_t_p = 0.0
            eps_i = 0.0
        
        # Calcul des distances L2 entre patches
        y_patches = extract_centered_patches(y_t, patchsize)
        dists = torch.sum((y_patches - Z)**2, dim=1) # Forme: (N_imgs, H*W) (4.1)

        # Soft argmin avec un Softmax (eq9, eq10)
        sm = F.softmax(-dists / (2 * tau_t), dim=0)  # Forme: (N_imgs, H*W) (4.2)
        
        # Reconstruction locale combinée (eq8)
        z_t = torch.sum(Z_center * sm.unsqueeze(1), dim=0)  
        z_t = z_t.view(1, C, H, W) # ravoir la bonne dimension/forme
        
        # Màj de y_t-1
        y_t = (1.0 - lambda_t) * y_t + lambda_t * z_t + sigma_t_p * eps_i # 5
        
        saved_steps.append(y_t.cpu())
        
    return saved_steps