from pytorch_lightning import LightningModule
from torch import Tensor, nn
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import torch
import math
import sys
from utils import set_weight_decay_per_param


class BaseModel(ABC, LightningModule):
    """
        Base model for Self-Supervised Learning (SSL), Vision-Language (VL) or Language-Guided (LG) models.
        We expect any `BaseModel` to implement a features extractor.
    """

    def __init__(self, optim_kwargs: Dict):
        super().__init__()
        self.optim_kwargs = optim_kwargs
        #self.b = 0.25
       # self.automatic_optimization = False

    def configure_optimizers(self):
        # Combine shared + head + ortho loss weights
        optimizer = torch.optim.AdamW(
        set_weight_decay_per_param(
            self, weight_decay=self.optim_kwargs["weight_decay"]),
        lr=self.optim_kwargs["lr"])

        return optimizer          

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch, cur_ep = self.current_epoch, mode="asymetric")
        loss_dict = self.loss(outputs)
        loss = ( loss_dict['loss_fusion']+( loss_dict['loss_mod1'] + loss_dict['loss_mod2']) 
                            +loss_dict['loss_ortho_bmods'] + loss_dict['loss_ortho_mod1']+ loss_dict['loss_ortho_mod2'] ) 

        #loss = (loss - self.b).abs() + self.b
        self.log("train/loss_total", loss, on_epoch=True,sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(*batch, mode="validation")
        loss_dict = self.loss(outputs)
        val_loss = ( loss_dict['loss_fusion']+ ( loss_dict['loss_mod1'] + loss_dict['loss_mod2'])
                            +loss_dict['loss_ortho_bmods'] + loss_dict['loss_ortho_mod1']+ loss_dict['loss_ortho_mod2'] ) 
        #val_loss = (val_loss - self.b).abs() + self.b
        self.log("val/loss_total", val_loss, on_epoch=True)
        self.log_dict({"val_%s"%k: v for k, v in loss_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(*batch,mode="test")
        loss_dict = self.loss(outputs)
        test_loss = ( loss_dict['loss_fusion']+ ( loss_dict['loss_mod1'] + loss_dict['loss_mod2'])
                            + loss_dict['loss_ortho_bmods'] + loss_dict['loss_ortho_mod1']+ loss_dict['loss_ortho_mod2']) 
        #test_loss = (test_loss - self.b).abs() + self.b
        self.log_dict({"test_%s"%k: v for k, v in loss_dict.items()}, on_epoch=True, sync_dist=True)
        return test_loss

    @abstractmethod
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs) \
            -> Tuple[Tensor, Tensor]:
        """
        Extract global average pooled visual features.
        Args:
            loader: Dataset loader to serve ``(image, label)`` tuples
        Returns:
            Pair (X,y) corresponding to extracted features and corresponding labels
        """
        pass
