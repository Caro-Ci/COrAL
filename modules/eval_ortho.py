import torch
import torch.nn.functional as F

@torch.no_grad()
def cosine_u1_u2(u1, u2):
    u1 = u1.float(); u2 = u2.float()
    c = F.cosine_similarity(u1, u2, dim=-1)   # [N], per-sample cos
    return c.abs().mean().item()

@torch.no_grad()
def grassmann_distance(X, Y, k=10):
    X = X.float(); Y = Y.float()
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Vx = torch.linalg.svd(Xc, full_matrices=False).Vh   
    Vy = torch.linalg.svd(Yc, full_matrices=False).Vh
    k = min(k, Vx.shape[0], Vy.shape[0])
    Qx, Qy = Vx[:k], Vy[:k]                              
    s = torch.linalg.svd(Qx @ Qy.T, full_matrices=False).S.clamp(-1, 1)  
    return torch.sin(torch.arccos(s)).mean().item()

