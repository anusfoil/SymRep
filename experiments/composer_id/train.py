import sys
sys.path.extend(["../../symrep", "../..", "../../model"])
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader, random_split
from torchmetrics import Accuracy, AUROC, F1Score
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning import loggers as pl_loggers
from einops import rearrange, reduce, repeat

from symrep import dataloaders, matrix
from model.frontend import cnn_baseline
from model.backend import baseline



class LitModel(LightningModule):
    def __init__(self, model_frontend, model_backend, cfg):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_frontend = model_frontend
        self.model_backend = model_backend
        self.n_classes = model_backend.n_classes

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()

        self.train_fscore = F1Score(num_classes=self.n_classes, average="macro")
        self.train_auroc = AUROC(num_classes=self.n_classes, average="macro")
        self.val_fscore = F1Score(num_classes=self.n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=self.n_classes, average="macro")

    def batch_to_input(self, batch):
        """convert a batch of data into with the designated representation format.
        """
        if self.cfg.experiment.symrep == "matrix":
            _input, _label = matrix.batch_to_matrix(batch, self.cfg)  
            
        return _input, _label


    def forward_pass(self, _input):
        """the same forward pass in both training_step and validation_step"""
        
        _input = rearrange(_input, "b s c h w -> (b s) c h w") 
        _seg_emb = rearrange(self.model_frontend(_input), "(b s) v -> b s v", s=self.cfg.experiment.n_segs) 
        _logits = self.model_backend(_seg_emb) # b n
        _pred = F.softmax(_logits, dim=1).argmax(dim=1)   
        return _logits, _pred     

    def training_step(self, batch):
        
        _input, _label = self.batch_to_input(batch)
        _logits, _pred = self.forward_pass(_input)

        loss = self.train_loss(_logits, _label)
        self.train_acc(_pred, _label)
        self.train_fscore(_pred, _label)
        self.train_auroc(_logits, _label)

        self.log('train_loss', loss.item(), prog_bar=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        self.log("train_fscore", self.train_fscore, on_epoch=True, sync_dist=True)
        self.log("train_auroc", self.train_auroc, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        _input, _label = self.batch_to_input(batch)
        _logits, _pred = self.forward_pass(_input) 

        loss = self.val_loss(_logits, _label)
        self.val_acc(_pred, _label)
        self.val_fscore(_pred, _label)
        self.val_auroc(_logits, _label)

        self.log('val_loss', loss.item(), prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.experiment.lr)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
            #     "interval": "epoch"
            # }
        }

class LitDataset(LightningDataModule):
    def __init__(self, cfg):
        super(LitDataset, self).__init__()
        self.cfg = cfg
        if cfg.experiment.dataset == "ASAP":
            asap_dataset = dataloaders.ASAP(cfg)
            label_encoder = asap_dataset.label_encoder
            self.n_classes = label_encoder.vocab_size

            train_len = int(len(asap_dataset)*0.8)
            self.train_set, self.valid_set = random_split(asap_dataset, [train_len, len(asap_dataset) - train_len])


    def train_dataloader(self):
        return DataLoader(self.train_set, 
                            batch_size=self.cfg.experiment.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_set, 
                            batch_size=self.cfg.experiment.batch_size)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: OmegaConf) -> None:

    lit_dataset = LitDataset(cfg)
    lit_model = LitModel(
        cnn_baseline.CNN(), 
        baseline.Aggregator(lit_dataset.n_classes),
        cfg)

    trainer = Trainer(
        accelerator="gpu",
        devices="1",
        max_epochs=cfg.experiment.epoch,
        logger=pl_loggers.WandbLogger(
                project="symrep",
                name=f"{cfg.experiment.task}-{cfg.experiment.dataset}"
            )
        )
    trainer.fit(lit_model, datamodule=lit_dataset)


if __name__ == "__main__":
        main()