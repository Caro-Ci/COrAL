from torch import nn
import torch.nn.functional as F
import torch
import copy
from collections import OrderedDict
from typing import Dict, List
# Local imports
from modules.base import BaseModel
from modules.coral_loss import  CoralLoss
from modules.mmfusion import MMFusion


class COrAL(BaseModel):
    """ self-supervised Contrastive Orthogonalized framework with 
        asymmetric masking for learning multimodal representations
    """

    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 uni_projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict,
                 asym_masking):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(COrAL, self).__init__(optim_kwargs)
        #self.automatic_optimization = False
        # create the encoder
        self.encoder = encoder

        self.head = projection
        
        self.uni_head_1 = uni_projection
        self.uni_head_2 = copy.deepcopy(uni_projection)

        # Build the loss
        self.loss = CoralLoss(**loss_kwargs)
        self.asym_masking = asym_masking
        self.ratio = 0.05
        


    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

       
        
    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], mode="asymetric", cur_ep =0):
        # compute features for all modalities
        all_masks =[[True,True]]
        if self.asym_masking:
            if cur_ep == 25:
               self.ratio= 0.35
            if cur_ep == 50:
                self.ratio= 0.55
            if cur_ep == 75:
                self.ratio= 0.75
            if mode == "asymetric":
                z1, z1mods = self.encoder(x1, mask_modalities=all_masks, masking=0,ratio = self.ratio)
                z2, z2mods = self.encoder(x2, mask_modalities=all_masks, masking=1,ratio= self.ratio)
            else:
                z1, z1mods = self.encoder(x1, mask_modalities=all_masks, masking=None)
                z2, z2mods = self.encoder(x2, mask_modalities=all_masks, masking=None)
        else:
            z1, z1mods = self.encoder(x1, mask_modalities=all_masks, masking=None)
            z2, z2mods = self.encoder(x2, mask_modalities=all_masks, masking=None)
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        #modality 1 at pos 0 for both augmentations
        z1mods[0] = self.uni_head_1(z1mods[0])
        z2mods[0] = self.uni_head_1(z2mods[0])
        #modality 2 at pos 1 for both augmentations
        z1mods[1] = self.uni_head_2(z1mods[1])
        z2mods[1] = self.uni_head_2(z2mods[1])
        return {'aug1_embed': z1,
                'aug2_embed': z2,
                'mod_augs1': z1mods,
                'mod_augs2': z2mods}
        
  
    #used to get the representations for the lin probing evaluation
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal embedding from the encoder.
           Returns Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor): 
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                fusion,emb_mod = self.encoder(X_, **kwargs)
                #concatinate into singular embedding
                mod_cat = torch.cat(emb_mod, dim=1)
                output= torch.cat((fusion, mod_cat), dim=1) 
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)
    
    def extract_all_the_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal embeddings before final concatination from the encoder. Used for UMAP-plots or validation
           Returns (y, Z_sr, Z_u1, Z_u2) corresponding labels, shared information embedding and modality-unique embeddings
        """
        y, Fuse, Mod1, Mod2 = [], [], [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output without fusion
                fusion,emb_mod = self.encoder(X_, **kwargs)
                fusion = self.head(fusion)
                emb_mod[0] = self.uni_head_1(emb_mod[0])
                emb_mod[1] = self.uni_head_2(emb_mod[1])
                y.extend(y_.detach().cpu())
                Fuse.extend(fusion.detach().cpu())
                Mod1.extend(emb_mod[0].detach().cpu())
                Mod2.extend(emb_mod[1].detach().cpu())
        torch.cuda.empty_cache()
        return  torch.stack(y, dim=0).to(self.device),torch.stack(Fuse, dim=0).to(self.device), torch.stack(Mod1, dim=0).to(self.device), torch.stack(Mod2, dim=0).to(self.device)