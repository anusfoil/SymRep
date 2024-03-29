import os, sys, random, glob, json
sys.path.extend(["../symrep", "../", "../model", "../converters"])
import hydra, wandb, scipy
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torchmetrics import Accuracy, AUROC, F1Score
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
from einops import rearrange, reduce, repeat

import dgl
from dgl.dataloading import GraphDataLoader
from converters import dataloaders, matrix, sequence, graph, utils
from model import agg, matrix_frontend, graph_frontend, sequence_frontend
import train_utils as train_utils


class LitModel(LightningModule):
    def __init__(self, model_frontend, model_backend, cfg):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_frontend = model_frontend
        self.model_backend = model_backend
        self.n_classes = model_backend.n_classes

        if cfg.experiment.symrep == "sequence":
            self.tokenizer = utils.construct_tokenizer(cfg)

        self.all_pred = []   # for testing and anova
        self.all_label = []

        self.train_acc = Accuracy('multiclass', num_classes=self.n_classes)
        self.val_acc = Accuracy('multiclass', num_classes=self.n_classes)
        self.test_acc = Accuracy('multiclass', num_classes=self.n_classes)
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_loss = torch.nn.CrossEntropyLoss()

        self.train_fscore = F1Score('multiclass', num_classes=self.n_classes, average="macro")
        self.train_auroc = AUROC('multiclass', num_classes=self.n_classes, average="macro")
        self.val_fscore = F1Score('multiclass', num_classes=self.n_classes, average="macro")
        self.val_auroc = AUROC('multiclass', num_classes=self.n_classes, average="macro")
        self.test_fscore = F1Score('multiclass', num_classes=self.n_classes, average="macro")
        self.test_auroc = AUROC('multiclass', num_classes=self.n_classes, average="macro")
        
        self.mem_count = []

    def batch_to_input(self, batch):
        """convert a batch of data into with the designated representation format.
        """
        if self.cfg.experiment.symrep == "matrix":
            _input, _label = matrix.batch_to_matrix(batch, self.cfg, self.device)  
        elif self.cfg.experiment.symrep == "sequence":
            _input, _label = sequence.batch_to_sequence(batch, self.cfg, self.device, self.tokenizer)  
        elif self.cfg.experiment.symrep == "graph":
            _input, _label = graph.batch_to_graph(batch, self.cfg, self.device)              
        return _input, _label


    def forward_pass(self, _input):
        """the same forward pass in both training_step and validation_step
        
        Graph: forward pass keeps each piece of data with different number of subgraphs without padding. With 
                uneven number of segments, they pass through backend individually.
        """
    
        if self.cfg.experiment.symrep == "graph":
            batch_n_segs = [len(data) for data in _input]
            _input = dgl.batch(np.concatenate(_input)).to(self.device)
            _seg_emb = torch.split((self.model_frontend(_input)), batch_n_segs)
            _logits = rearrange([
                self.model_backend(rearrange(seg, "s v -> 1 s v")) for seg in _seg_emb],
                "b 1 n -> b n")
        else:
            n_segs = _input.shape[1]
            if self.cfg.experiment.symrep == "matrix":
                _input = rearrange(_input, "b s c h w -> (b s) c h w") 
            elif self.cfg.experiment.symrep == "sequence":
                self.seg_len = _input.shape[1]
                if self.cfg.sequence.mid_encoding == "CPWord":
                    _input = rearrange(_input, "b s l k -> (b s) l k") #CPWord has one extra batch dimension
                else:
                    _input = rearrange(_input, "b s l -> (b s) l") 

            if self.cfg.sequence.output_weights:
                _output, weights = self.model_frontend(_input)
                
                self._input = _input
                self.weights = weights
                return None, None
            
            _seg_emb = rearrange(self.model_frontend(_input), "(b s) v -> b s v", s=n_segs) 
            _logits = self.model_backend(_seg_emb) # b n
        
        _pred = F.softmax(_logits, dim=1).argmax(dim=1)   
        return _logits, _pred     

    def training_step(self, batch):

        # if self.current_epoch == 1: # log the dataset to wandb after the first epoch of computing
        #     data_save_dir = self.cfg.experiment.data_save_dir[28:] # remove my own root dir path
        #     data_artifact = wandb.Artifact("processed_input", type="data")
        #     data_artifact.add_dir(self.cfg.experiment.data_save_dir, name=data_save_dir)
        #     self.logger.experiment.log_artifact(data_artifact)

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


    def test_step(self, batch, batch_idx):
        
        _input, _label = self.batch_to_input(batch)
        _logits, _pred = self.forward_pass(_input) 

        if self.cfg.sequence.output_weights:
            weights, adjs = [], []
            for i in range(len(batch[0])):
                i = 3
                print(batch[0][i])
                piece_idx = i
                seg_len = _input.shape[1]
                return_flag = False
                n_nodes = train_utils.attn_visualization(self._input, self.weights, self.tokenizer, seg_len, i, 
                                                                        piece_idx=piece_idx, token_type="NoteOn", 
                                                                        subseq_len=65,
                                                                        return_attn=return_flag)
                # get the graph for comparison
                adj = train_utils.graph_visualization(self.cfg, batch[0][piece_idx], n_nodes, i, return_adj=return_flag)
                # weights.append(attn_weights.flatten())
                # adjs.append(adj.flatten())
                
            # scipy.stats.pearsonr(torch.cat(weights), torch.cat(adjs))

        self.all_pred.extend(_pred.cpu().tolist())
        self.all_label.extend(_label.cpu().tolist())

        self.test_acc(_pred, _label)
        self.test_fscore(_pred, _label)
        self.test_auroc(_logits, _label)

        self.log('test_acc', self.test_acc, prog_bar=True, sync_dist=True)
        self.log("test_fscore", self.test_fscore, sync_dist=True)
        self.log("test_auroc", self.test_auroc, sync_dist=True)
        return 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.experiment.lr, weight_decay=1e-2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.experiment.lr_gamma),
                "interval": "epoch"
            }
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
            #     "interval": "epoch",
            #     "frequency": 1,
            #     "monitor": "val_fscore"
            # }
        }

class LitDataset(LightningDataModule):
    def __init__(self, cfg):
        super(LitDataset, self).__init__()
        # self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = cfg.experiment.batch_size
        if cfg.experiment.dataset == "ASAP":
            dataset = dataloaders.ASAP(cfg)
        elif cfg.experiment.dataset == "ATEPP":
            dataset = dataloaders.ATEPP(cfg)

        self.dataset = dataset
        label_encoder = dataset.label_encoder
        self.n_classes = label_encoder.vocab_size

        kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=cfg.experiment.random_seed)
        folds_generator = kf.split(X=range(len(dataset)), y=dataset.label_column)
        folds = [next(folds_generator) for _ in range(8)]
        self.train_indices, self.test_indices = folds[cfg.experiment.fold_idx]

        # train_indices, test_indices = train_test_split((range(len(dataset))), 
        #                                                test_size=cfg.experiment.test_split_size, 
        #                                                stratify=dataset.label_column,
        #                                                random_state=cfg.experiment.random_seed) 
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)


    def train_dataloader(self):
        return DataLoader(self.dataset, 
                            sampler=self.train_sampler,
                            # sampler=self.train_indices,
                            batch_size=self.batch_size,
                            # pin_memory=True,
                            num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, 
                            sampler=self.test_sampler,
                            # sampler=self.test_indices,
                            batch_size=self.batch_size,
                            # pin_memory=True,
                            num_workers=8)


def prep_run(cfg):
    """random seed, empty cache, clean wandb logs"""
    torch.manual_seed(cfg.experiment.random_seed)
    random.seed(cfg.experiment.random_seed)
    torch.use_deterministic_algorithms(True)

    torch.cuda.empty_cache()
    
    os.system("wandb sync --clean-force --clean-old-hours 3") # clean the wandb local outputs....

    return 


def construct_model_frontend(cfg, lit_dataset):
    """set the frontend based on the symbolic representation."""
    if cfg.experiment.symrep == "matrix":
        model = matrix_frontend.Resnet(cfg)
        # model = matrix_frontend.CNN(cfg)
    elif cfg.experiment.symrep == "sequence":
        model = sequence_frontend.AttentionEncoder(cfg)
    elif cfg.experiment.symrep == "graph":
        """try getting one graph and see their input dimension to give to model"""
        if cfg.experiment.input_format == "musicxml":
            g = graph.musicxml_to_graph(lit_dataset.dataset[0][0], cfg)
        else:
            g = graph.perfmidi_to_graph(lit_dataset.dataset[0][0], cfg)
        in_dim = g.ndata['feat_0'].shape[1] 
        if cfg.experiment.feat_level:
            in_dim += g.ndata['feat_1'].shape[1] 
        model = graph_frontend.GNN(cfg, in_dim=in_dim)

    return model


def construct_logger(cfg, model):
    wandb_logger = pl_loggers.WandbLogger(
                project="symrep",
                name=cfg.experiment.exp_name,
                group=(cfg.experiment.group_name if cfg.experiment.grouped else None),
                log_model="all"
            )
    # wandb_logger.watch(model, log='all')
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    return wandb_logger


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: OmegaConf) -> None:

    if cfg.experiment.symrep == 'graph': 
        cfg.experiment.group_name="${experiment.task}-${experiment.dataset}-${experiment.input_format}-${experiment.symrep}-bi${graph.bi_dir}-ho${graph.homo}-noae"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.device)
    prep_run(cfg) # random seed, empty cache, clean wandb logs

    lit_dataset = LitDataset(cfg)
    model_frontend = construct_model_frontend(cfg, lit_dataset)
    lit_model = LitModel(
        model_frontend, 
        agg.AttentionAggregator(cfg, lit_dataset.n_classes),
        cfg)
    
    wandb_logger = construct_logger(cfg, model_frontend) # wandblogger, model watch and artifact dataset
    checkpoint_callback = ModelCheckpoint(
                                monitor="val_acc",
                                dirpath=cfg.experiment.checkpoint_dir,
                                filename='{epoch:02d}-{val_acc:.2f}',
                                mode='max'
                            )
    trainer = Trainer(
        accelerator="gpu",
        # gpus=cfg.experiment.device,
        gpus=0,
        # strategy='ddp',
        max_epochs=cfg.experiment.epoch,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        # auto_scale_batch_size='power',
        callbacks=[
            checkpoint_callback, 
            EarlyStopping(
                monitor="val_acc",
                min_delta=cfg.experiment.es_threshold,
                # patience=300,  
                patience=cfg.experiment.es_patience,  # NOTE no. val epochs, not train epochs
                verbose=False,
                mode="max",
            ),
            LearningRateMonitor(logging_interval='epoch')
            ]
        )
    
    if (not cfg.sequence.output_weights) and (not cfg.experiment.testing_only):
        trainer.fit(lit_model, 
            datamodule=lit_dataset,
            ckpt_path=(cfg.experiment.checkpoint_file if cfg.experiment.load_model else None))


    if cfg.experiment.load_model:
        artifact_dir = wandb_logger.download_artifact(artifact=cfg.experiment.artifact_dir)
        wandb_logger.use_artifact(artifact=cfg.experiment.artifact_dir)

    """testing"""
    if cfg.experiment.dataset == "ASAP":
        test_dataset = dataloaders.ASAP(cfg, split='test')
    elif cfg.experiment.dataset == "ATEPP":
        test_dataset = dataloaders.ATEPP(cfg, split='test')
    trainer.test(lit_model, 
                 ckpt_path=(artifact_dir+"/model.ckpt" if cfg.experiment.load_model else None ),
                #  callbacks = [checkpoint_callback],
                 dataloaders=DataLoader(test_dataset, 
                                                batch_size=cfg.experiment.batch_size,
                                                pin_memory=True,
                                                num_workers=1,
                                                shuffle=True))
    print(lit_model.all_pred)
    print(lit_model.all_label)


if __name__ == "__main__":

    main()