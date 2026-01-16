import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad




class CoralLoss(nn.Module):

    def __init__(self, temperature=0.1, ortho_weights=1, unique_weights=1, shared_weights=1):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.ortho_w = ortho_weights
        self.unique_w = unique_weights
        self.shared_w = shared_weights
        
    def orthogonality_loss(self, embeddings1, embeddings2):
        """
        Encourages modality embeddings to be orthogonal via CosineEmbeddingLoss.
        Assumes inputs are of shape [batch, dim].
        """
        batch = embeddings1.shape[0]
        target = -torch.ones(batch, device=embeddings1.device)  
        ortho_loss = nn.CosineEmbeddingLoss(reduction="mean")
        return ortho_loss(embeddings1, embeddings2, target)

    
    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature 
        sim_zjj = (z2 @ z2.T) / self.temperature
        sim_zij = (z1 @ z2.T) / self.temperature 
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        return loss
    
    def forward(self, outputs):
       z1, z2, aug1, aug2 = outputs["aug1_embed"], outputs["aug2_embed"], outputs['mod_augs1'], outputs['mod_augs2']
       assert len(z1) == len(z2)
       n_emb = len(z1)
       n_em_mod = len(aug1)
       z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
       z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
       Z = all_gather_batch_with_grad(z1 + z2)
       z1, z2 = Z[:n_emb], Z[n_emb:]
       z1 = torch.stack(z1)
       z2 = torch.stack(z2)
       aug1 = [func.normalize(z, p=2, dim=-1) for z in aug1]
       aug2 = [func.normalize(z, p=2, dim=-1) for z in aug2]
       Z_mod = all_gather_batch_with_grad(aug1 + aug2)
       aug1, aug2 = Z_mod[:n_em_mod], Z_mod[n_em_mod:]

       loss_fusion = self.infonce(z1[-1], z2[-1])
       loss_mod1 = self.infonce(aug1[0], aug2[0])
       loss_mod2 = self.infonce(aug1[1], aug2[1])

       loss_ortho_bmods =( self.orthogonality_loss(aug1[0], aug1[1]) +self.orthogonality_loss(aug2[0], aug2[1]))
       loss_ortho_mod1= (self.orthogonality_loss(aug1[0], z1[-1]) + self.orthogonality_loss(aug2[0], z2[-1]))
       loss_ortho_mod2 =(self.orthogonality_loss(aug1[1], z1[-1]) + self.orthogonality_loss(aug2[1], z2[-1]))

       loss_ortho_bmods =self.ortho_w * loss_ortho_bmods
       loss_ortho_mod1=self.ortho_w*loss_ortho_mod1
       loss_ortho_mod2=self.ortho_w*loss_ortho_mod2

       loss_fusion = self.shared_w * loss_fusion 
       loss_mod1 = self.unique_w * loss_mod1
       loss_mod2 = self.unique_w * loss_mod2
   
       return {
           "loss_fusion": loss_fusion,
           "loss_mod1": loss_mod1,  
           "loss_mod2": loss_mod2,
           "loss_ortho_bmods": loss_ortho_bmods,
           "loss_ortho_mod1": loss_ortho_mod1,
           "loss_ortho_mod2": loss_ortho_mod2
       }

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)

