from cProfile import label
import os
from random import shuffle
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, auroc
from transformers import BertModel, BertTokenizer
import re

from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type
DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "data")

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    f = re.compile(r"<[^>]*?>|&amp;|[/'’\"”]")
    data["description"] = data["description"].map(lambda x: f.sub(" ", x))
    #data["description"] = data["description"].map(lambda x: x.lstrip())
    return data

class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        id_name: str,
        description_name: str,
        jobflag_name: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.id_name = id_name
        self.description_name = description_name
        self.jobflag_name = jobflag_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        id = data_row[self.id_name]
        description = data_row[self.description_name]
        jobflag_name = data_row[self.jobflag_name]

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(jobflag_name),
        )


class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame = None, 
        #valid_df: pd.DataFrame, 
        test_df: pd.DataFrame = None, 
        train_fold: pd.DataFrame = None,
        val_fold: pd.DataFrame = None,
        batch_size: int = None, 
        max_token_len: int = None, 
        id_column_name: str = 'id',
        description_column_name: str = 'description',
        jobflag_column_name: str = 'jobflag',
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.train_df = train_df
        #self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.id_column_name = id_column_name
        self.description_column_name = description_column_name
        self.jobflag_column_name = jobflag_column_name
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage):
        self.train_dataset = AugmentedDataset(
            self.train_df, 
            self.tokenizer, 
            self.max_token_len,
            self.id_column_name,
            self.description_column_name,
            self.jobflag_column_name
        )

        self.test_dataset = AugmentedDataset(
            self.test_df, 
            self.tokenizer, 
            self.max_token_len,
            self.id_column_name,
            self.description_column_name,
            self.jobflag_column_name
        )
    
    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_fold, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

class EnsembleVotingModel(pl.LightningModule):
    def __init__(self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        #self.test_acc = Accuracy()

    def test_step(self, batch, batch_idx):
        preds, output = self.forward(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"])
        preds_all = torch.stack([m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) for m in self.models]).mean(0)
        loss = self.criterion(preds_all, batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def test_epoch_end(self, outputs, mode="test_all"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)
        
        epoch_auroc = auroc(epoch_preds, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)  

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

class BertRanker(pl.LightningModule):
    def __init__(
        self, 
        n_classes: int, 
        n_linears: int,
        d_hidden_linear: int,
        dropout_rate: float,
        learning_rate: float,
        pooling_type: str,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        
        if pooling_type == '4_cls':
            classifier_hidden_size = self.bert.config.hidden_size * 4
        else:
            classifier_hidden_size = self.bert.config.hidden_size

        if n_linears == 1:
            self.classifier = nn.Linear(classifier_hidden_size, n_classes)
        else:
            classifier = nn.Sequential(
                nn.Linear(classifier_hidden_size, d_hidden_linear),
                nn.Sigmoid(),
                nn.Dropout(p=dropout_rate),
            )
            for i in range(n_linears-1):
                classifier.add_module('fc_{}'.format(i), nn.Linear(d_hidden_linear, d_hidden_linear))
                classifier.add_module('activate_{}'.format(i), nn.Sigmoid())
                classifier.add_module('dropout_{}'.format(i), nn.Dropout(p=dropout_rate))
            classifier.add_module('fc_last', nn.Linear(d_hidden_linear, n_classes))
            self.classifier = classifier
        
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.pooling_type = pooling_type

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        if self.pooling_type == 'cls':
            cls = output.pooler_output
            preds = self.classifier(cls)
        if self.pooling_type == 'max':
            mp = output.last_hidden_state.max(1)[0]
            preds = self.classifier(mp)
        if self.pooling_type == '4_cls':
            clses = torch.cat([output.hidden_states[-1*i][:,0] for i in range(1, 4+1)], dim=1)
            preds = self.classifier(clses)
        preds = self.classifier(output.pooler_output)
        preds = torch.flatten(preds)
        return preds, output
      
    def training_step(self, batch, batch_idx):
        preds, output = self.forward(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"])
        loss = self.criterion(preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def validation_step(self, batch, batch_idx):
        preds, output = self.forward(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"])
        loss = self.criterion(preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        preds, output = self.forward(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"])
        loss = self.criterion(preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}
    
    def training_epoch_end(self, outputs, mode="train"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)
        
        epoch_auroc = auroc(epoch_preds, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)                  

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)
        
        epoch_auroc = auroc(epoch_preds, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)                    

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

def make_callbacks(min_delta, patience, checkpoint_path):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.pl_seed, workers=True)
    cwd = hydra.utils.get_original_cwd()
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )
    wandb_logger.log_hyperparams(cfg)
    #data = pd.read_pickle(cfg.path.data_file_name)
    #train, test = train_test_split(data, test_size=cfg.training.test_size, shuffle=True)
    #train, valid = train_test_split(train, test_size=cfg.training.valid_size, shuffle=True)
    #train = pd.read_pickle(cfg.path.train_file_name)
    #test = pd.read_pickle(cfg.path.test_file_name)
    train_df = pd.read_csv(cfg.path.train_file_path)
    test_df = pd.read_csv(cfg.path.test_file_path)
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    data_module = CreateDataModule(
        train_df=train_df,
        test_df=test_df,
        batch_size=cfg.training.batch_size,
        max_token_len=cfg.model.max_token_len,
    )
    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = BertRanker(
        n_classes=cfg.model.n_classes,
        n_linears=cfg.model.n_linears,
        d_hidden_linear=cfg.model.d_hidden_linear,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=cfg.training.learning_rate,
        pooling_type=cfg.model.pooling_type,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices=cfg.training.n_gpus,
        accelerator="gpu",
        #progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
        deterministic=True
    )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="./")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, data_module)
                           
    #data_module.setup(stage='test')                       
    #results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    #print(results)

if __name__ == "__main__":
    main()
