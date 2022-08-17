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
from sklearn.model_selection import KFold, StratifiedKFold


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    f = re.compile(r"<[^>]*?>|&amp;|[/'’\"”]")
    data["description"] = data["description"].map(lambda x: f.sub(" ", x))
    #data["description"] = data["description"].map(lambda x: x.lstrip())
    return data


def get_kfold(train, n_splits, seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    generator = kf.split(train)
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def get_stratifiedkfold(train, target_col, n_splits, seed):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    generator = kf.split(train, train[target_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        id_name: str,
        description_name: str,
        jobflag_name: str = None,
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
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None, 
        batch_size: int = None, 
        max_token_len: int = None, 
        id_column_name: str = 'id',
        description_column_name: str = 'description',
        jobflag_column_name: str = 'jobflag',
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.id_column_name = id_column_name
        self.description_column_name = description_column_name
        self.jobflag_column_name = jobflag_column_name
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage = None) -> None:
        if stage == 'fit':
            self.train_dataset = AugmentedDataset(
                self.train_df, 
                self.tokenizer, 
                self.max_token_len,
                self.id_column_name,
                self.description_column_name,
                self.jobflag_column_name
            )

            self.valid_dataset = AugmentedDataset(
                self.valid_df, 
                self.tokenizer, 
                self.max_token_len,
                self.id_column_name,
                self.description_column_name,
                self.jobflag_column_name
            )

        if stage == 'test':
            self.test_dataset = AugmentedDataset(
                self.test_df, 
                self.tokenizer, 
                self.max_token_len,
                self.id_column_name,
                self.description_column_name,
                self.jobflag_column_name
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


class Classifier(pl.LightningModule):
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
    #checkpoint_path = os.path.join(
    #    wandb_logger.experiment.dir, cfg.path.checkpoint_path
    #)
    wandb_logger.log_hyperparams(cfg)

    train_df = pd.read_csv(cfg.path.train_file_name)
    #train_df = train_df.rename(columns={'jobflag': 'label'})
    test_df = pd.read_csv(cfg.path.test_file_name)
    test_df["jobflag"] = 0
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    trn_fold = [0, 1, 2, 3, 4]
    if cfg.training.fold == 'kf':
        folds = get_kfold(train=train_df, n_splits=5, seed=cfg.training.pl_seed)
    if cfg.training.fold == 'skf':
        folds = get_stratifiedkfold(train=train_df, target_col='jobflag', n_splits=5, seed=cfg.training.pl_seed)

    test_acc_all = []
    test_auroc_all = []
    for fold in trn_fold:
        train_df_fold, valid_df_fold = train_df.loc[folds!=fold], train_df[folds==fold]
        checkpoint_path = os.path.join(
            wandb_logger.experiment.dir, cfg.path.checkpoint_path + '_{}'.format(fold)
        )
        data_module = CreateDataModule(
            train_df=train_df_fold,
            valid_df=valid_df_fold,
            test_df=test_df,
            batch_size=cfg.training.batch_size,
            max_token_len=cfg.model.max_token_len,
        )
        data_module.setup(stage='fit')

        call_backs = make_callbacks(
            cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
        )

        model = Classifier(
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
            deterministic=True,
            num_folds=5,
            shuffle=False,
            stratified=True,
        )
                
        data_module.setup(stage='test')                       
        results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
        print(results)
        test_acc_all.append(results[0]['test_accuracy'])
        test_auroc_all.append(results[0]['test_auroc'])
    print('test_acc_avg', sum(test_acc_all) / len(test_acc_all))
    print('test_auroc_avg', sum(test_auroc_all) / len(test_auroc_all))


if __name__ == "__main__":
    main()
