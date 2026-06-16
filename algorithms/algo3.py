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


@torch.no_grad()
def algo3_kwatra_modifie(D_train, patchsize=3, N=50, device='cpu',seed=None):
    """
    Algorithme 3 : Algorithme de Kwatra modifié.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    D_train = D_train.to(device)
    N_imgs, C, H, W = D_train.shape
   
    # Pour chaque position i, y'a N_imgs patches
    Z = extract_centered_patches(D_train, patchsize) # (N_imgs, C*patchsize^2, H*W)
    
    # Extraction du pixel central de chaque patch d'origine
    center_p = (patchsize**2) // 2
    c_indices = [c * (patchsize**2) + center_p for c in range(C)]
    Z_center = Z[:, c_indices, :] # (N_imgs, C, H*W)
    
    # 1 z_0(i) = voisinage aléatoire
    # Stockage de l'index de l'image source
    z_n_indices = torch.randint(0, N_imgs, (H * W,), device=device) 
    
    saved_steps = []
    
    for n in tqdm(range(N)):
        # Pas de boucle 3 pour les pixels i car calcul vectoriel
        
        # 4 Le pixel prend la valeur du pixel central du patch assigné
        z_n_indices_expanded = z_n_indices.view(1, 1, H * W).expand(1, C, H * W)
        x_n1_center = torch.gather(Z_center, 0, z_n_indices_expanded) # (1, C, H*W)
        x_n1 = x_n1_center.view(1, C, H, W) # Reformer l'image
        
        saved_steps.append(x_n1.cpu())
        
        # Extraction des patches de la nouvelle image reconstruite
        x_patches = extract_centered_patches(x_n1, patchsize) # (1, C*patchsize^2, H*W)
        
        # Calcul des distances de position i de x_patches avec la position i dans Z
        dists = torch.sum((x_patches - Z)**2, dim=1) # (N_imgs, H*W)
        
        # 5    
        z_n1_indices = torch.argmin(dists, dim=0) # (H*W,)
        
        # 7 Si les patches assignés ne changent plus (onverge,ce)
        if torch.equal(z_n1_indices, z_n_indices):
            break
            
        z_n_indices = z_n1_indices

    return saved_steps