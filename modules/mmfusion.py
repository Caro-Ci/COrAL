import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import List, Optional, Union
from einops import repeat
from collections import OrderedDict


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))  
    
    def forward(self, x):  
        scores = torch.einsum('btd,d->bt', x, self.query)
        weights = F.softmax(scores, dim=1)  
        pooled = torch.einsum('btd,bt->bd', x, weights)
        return pooled



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, d_model: int, n_head: int,
                 dropout: float = 0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, add_bias_kv=False,
                                          dropout=dropout,  batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x.clone(), x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    


class FusionTransformer_new(nn.Module):
    """Fusion of features from multiple modalities, batch-first
    in_shape: (N, L1, E), (N, L2, E), out_shape: (N, E)
    pooling: cls, concatenation over tokens + self-attention for fusion
    """
    def __init__(self, width: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float = 0.):
        
        super().__init__()
        self.width = width
        self.layers = n_layers
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) 
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, n_heads, dropout=dropout)
            for _ in range(1)])
        
        self.initialize()

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: List[torch.Tensor]):
        """
        :param x: input tensors
        :return:
        """
        # Concatenate over tokens + self-attention
        x = torch.cat(x, dim=self.token_dim)
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=self.token_dim)
        for layer in self.resblocks:
            x = layer(x)
        x = self.norm(x)
        x = x[:, 0] if self.token_dim == 1 else x[0]
        return x



class MMFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 input_adapters: List[nn.Module],
                 lin_layers: List[nn.Module],
                 embed_dim: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 1,
                 dropout: float = 0.):
        """ Multi-Modal (MM) fusion model using FusionTransformer in the shared path and MLPs in the unique paths
        Each modality is encoded by shared and unique encoder, tokenized if needed and given to FusionTransformer and MLPs.
        Output is a shared and two unique embeddings for bimodal data. 

        encoders: List of Torch encoders (CNN, Transformer, MLP, etc.) for each modality
        input_adapters: List of Torch adapters for each modality for tokenization (can be None if not required)
        embed_dim: Embedding size
        n_heads: Number of heads in  multi-heads attention blocks (shared path)
        n_layers: Number of attention layers in latent fusion (shared path)
        dropout: attention matrix dropout rate (shared path)
        """
        super().__init__()
        assert len(encoders) == len(input_adapters), "Each encoder must have an adapter."
        self.input_adapters = nn.ModuleList(input_adapters)
        self.pooling_layers = nn.ModuleList([AttentionPooling(dim=int(embed_dim/2)) for _ in range(2)])
        self.encoders = nn.ModuleList(encoders)
        self.unique_encoders =nn.ModuleList([copy.deepcopy(encoder) for encoder in encoders])
        self.unique_input_adapters =nn.ModuleList([copy.deepcopy(adapter) for adapter in input_adapters])
        self.lin_mod =nn.ModuleList(lin_layers) 
        self.num_modalities = len(self.encoders)
        self.fusion_transformer = FusionTransformer_new(embed_dim, n_heads, n_layers, dropout)

    def apply_asymmetric_mask(self, z_tokens, masking, ratio):
        assert 0 <= masking < len(z_tokens)
        z_masked: List[torch.Tensor] = []
        for idx, z in enumerate(z_tokens):
            if idx == masking:
                B, T, D = z.shape
                # random boolean mask [B, T], True = keep, False = drop
                keep = torch.rand(B, T, device=z.device) >  ratio
                # expand to [B, T, D] for elementwise multiply
                keep = keep.unsqueeze(-1).float()
                z_masked.append(z * keep)
            else:
                # leave other modality unchanged
                z_masked.append(z)
        
        return z_masked  
        
        
    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None,masking: Optional[int] = None,ratio: Optional[float] = None):
        """
        x: List of tensors
        mask_modalities: Mask indicating which modalities are given. By default, `x` should have all modalities.
        masking: Indicator for which modality the asymmetric masking should be applied, if None, no masking is applied.
        ratio: percentage of masking for the asymmetric masking.
        :return: a latent vector z or list of vector if `mask_modalities` is a list of list.
        """
        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]

        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")

        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")
        #making sure modalities, encoders and adapters are all there
        encoders =self.encoders
        unique_encoders= self.unique_encoders
        input_adapters = self.input_adapters
        unique_input_adapters = self.unique_input_adapters
        # 1. Encode input modalities
        z = []
        for i, (enc, xi)  in enumerate(zip(encoders, x)):
            embedding = enc(xi)
            z.append(embedding)

        # unique encoders:
        z_uni = []
        for i, (enc_u, xi)  in enumerate(zip(unique_encoders, x)): 
            embedding_u = enc_u(xi)
            if isinstance(embedding_u, dict): 
                embedding_u = embedding_u["token_embeddings"]
            z_uni.append(embedding_u)
        # 2. Tokenize each latent features
        latent_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(input_adapters, z)]
       
        if masking is not None:
            latent_tokens = self.apply_asymmetric_mask(
                latent_tokens,masking=masking, ratio=ratio)
        
        uni_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(unique_input_adapters, z_uni)]

        # 3. Fusion
        if list_mask_mod is None:
            z = self.fusion_transformer(latent_tokens)
        else:
            z = []
            for mask_mod in list_mask_mod:
                latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
                # 3. FusionTransformer forward pass
                z.append(self.fusion_transformer(latent_tokens_))
        #modalities reduce dimension
        # lin layer
        uni_tokens = [lini(zi) for (lini, zi) in zip(self.lin_mod, uni_tokens)]
        uni_embeddings = [pool(xi) for (pool, xi) in zip(self.pooling_layers, uni_tokens)]
        return  z, uni_embeddings
    
