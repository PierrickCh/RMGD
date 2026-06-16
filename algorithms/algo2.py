import torch
import torch.nn as nn
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
def algo2_kwatra_original(D_train, patchsize=3, N=100, device='cpu',seed=None):
    """
    Algorithme 2 : Algorithme de Kwatra original.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    D_train = D_train.to(device)
    N_imgs, C, H, W = D_train.shape
    pad = patchsize // 2
    
    # Extraction des patches fixe Z de l'image initiale
    Z_unfold = extract_centered_patches(D_train, patchsize) # (N_imgs, C*p^2, H*W) avec p = patchsiez mais plus court
    Z = Z_unfold.transpose(1, 2)#.reshape(-1, C * patchsize**2) # (N_imgs, H*W, C*p^2)
    
    # Fold pour l'étape 4 (moyennage des patches), un peu comme Patch_Average de NIFTY
    fold = nn.Fold(output_size=(H, W), kernel_size=patchsize, padding=pad, stride=1)
    
    # Matrice pour normaliser l'addition des patches superposés (l'astuce de nifty expliquée à la réunion)
    ones_patches = torch.ones(1, C * patchsize**2, H * W, device=device)
    count = fold(ones_patches)
    count= (count+1.*(count==0)) 

    # 1 z_0(i) = voisinage aléatoire dans Z
    n_patches_dict = Z.shape[0]
    z_n_indices = torch.randint(0, n_patches_dict, (1, H * W, 1), device=device)
    
    saved_steps = []
    for n in tqdm(range(N)):
        # Pas de boucle pour 3 car calcul vectoriel
        # 4 
        
        z_n_patches=torch.gather(Z,0,z_n_indices.expand(-1,-1,C*patchsize**2)).permute(0,2,1) # (1, C*p^2, H*W)

        x_n_plus_1 = fold(z_n_patches) / count # Moyennage des patches se chevauchants 
        
        saved_steps.append(x_n_plus_1.clone().cpu())

        # 5 
        x_patches = extract_centered_patches(x_n_plus_1, patchsize) # Extraction des patchs de la nouvelle image x_n+1 pour trouver les plus proches dans Z (1, C*p^2, H*W)
        x_flat = x_patches.squeeze(0).transpose(0, 1) # (H*W, C*p^2)
        
        # Calcul des distances entre les patches de x et Z
        dists = torch.cdist(x_flat.unsqueeze(0).permute(1,0,2), Z.permute(1,0,2), p=2.0) # Forme: ( H*W, 1,N_imgs)
        
        z_n_plus_1_indices = torch.argmin(dists, dim=-1, keepdim=True).transpose(0, 1) # (1, H*W, 1)

        # 7 Si y'a plus de changement, stop
        if torch.equal(z_n_plus_1_indices, z_n_indices):
            break
            
        z_n_indices = z_n_plus_1_indices

    return saved_steps