'''
add scheduler
deberta-v3-large
concat
CV=
LB=
'''
import collections
import os
import random
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
TEST_FILE = os.path.join(DATA_PATH, "test.csv")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
EXP_NUM = "exp_7"
MODELS_DIR = "./models/"
MODEL_NAME = 'deberta-v3-large'
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.1
NUM_CLASSES = 4
MAX_TOKEN_LEN = 512
D_HIDDEN_LINEAR = 128
#POOLING_TYPE = 'cls'
#POOLING_TYPE = 'max'
POOLING_TYPE = 'concat'
EPOCHS = 15
#FOLD_TYPE = 'kf'
FOLD_TYPE = 'skf'
NUM_SPLITS = 4


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

class MyDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer,
        max_token_len: int,
        id_name: str = 'id',
        description_name: str = 'description',
        jobflag_name: str = 'jobflag',
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

class Classifier(nn.Module):
    def __init__(
        self, 
        n_classes: int, 
        #n_linears: int,
        d_hidden_linear: int,
        dropout_rate: float,
        #learning_rate: float,
        pooling_type: str,
        pretrained_model=MODEL_NAME,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        
        if pooling_type == 'concat':
            classifier_hidden_size = self.bert.config.hidden_size * 4
        else:
            classifier_hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Linear(classifier_hidden_size, n_classes)
        
        #self.lr = learning_rate
        self.dropout = nn.Dropout(dropout_rate)
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
            preds = self.classifier(self.dropout(cls))
        if self.pooling_type == 'max':
            mp = output.last_hidden_state.max(1)[0]
            preds = self.classifier(self.dropout(mp))
        if self.pooling_type == 'concat':
            clses = torch.cat([output.hidden_states[-1*i][:,0] for i in range(1, 4+1)], dim=1)
            preds = self.classifier(self.dropout(clses))
        #preds = self.classifier(output.pooler_output)
        #preds = torch.flatten(preds)
        return preds, output
      
def train_fn(dataloader, model, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, total=len(dataloader))
    for i, batch in enumerate(progress):
        progress.set_description(f"<Train> Epoch{epoch+1}")
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask=batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        preds, _ = model.forward(input_ids=input_ids,
                                     attention_mask=attention_mask)
        loss = model.criterion(preds, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(preds, axis=1)
        total_loss += loss.item()
        total_corrects += torch.sum(preds == labels)

        all_labels += labels.tolist()
        all_preds += preds.tolist()

        progress.set_postfix(loss=total_loss/(i+1), f1=f1_score(all_labels, all_preds, average="macro"))

        train_loss = total_loss / len(dataloader)
        train_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average="macro")

    return train_loss, train_acc, train_f1

def eval_fn(dataloader, model, epoch):
    model.eval()
    total_loss = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))
        
        for i, batch in enumerate(progress):
            progress.set_description(f"<Valid> Epoch{epoch+1}")
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask=batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            preds, _ = model.forward(input_ids=input_ids,
                                     attention_mask=attention_mask)
            loss = model.criterion(preds, labels)

            preds = torch.argmax(preds, axis=1)
            total_loss += loss.item()
            total_corrects += torch.sum(preds == labels)

            all_labels += labels.tolist()
            all_preds += preds.tolist()

            progress.set_postfix(loss=total_loss/(i+1), f1=f1_score(all_labels, all_preds, average="macro"))

    valid_loss = total_loss / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = f1_score(all_labels, all_preds, average="macro")

    return valid_loss, valid_acc, valid_f1

def plot_training(train_losses, train_accs, train_f1s,
                  valid_losses, valid_accs, valid_f1s,
                  epoch, fold):
    
    loss_df = pd.DataFrame({"Train":train_losses,
                            "Valid":valid_losses},
                        index=range(1, epoch+2))
    loss_ax = sns.lineplot(data=loss_df).get_figure()
    loss_ax.savefig(f"./figures/loss_plot_fold={fold}.png", dpi=300)
    loss_ax.clf()

    acc_df = pd.DataFrame({"Train":train_accs,
                           "Valid":valid_accs},
                          index=range(1, epoch+2))
    acc_ax = sns.lineplot(data=acc_df).get_figure()
    acc_ax.savefig(f"./figures/acc_plot_fold={fold}.png", dpi=300)
    acc_ax.clf()

    f1_df = pd.DataFrame({"Train":train_f1s,
                          "Valid":valid_f1s},
                         index=range(1, epoch+2))
    f1_ax = sns.lineplot(data=f1_df).get_figure()
    f1_ax.savefig(f"./figures/f1_plot_fold={fold}.png", dpi=300)
    f1_ax.clf()

def trainer(fold, fold_indices, df):
    
    #train_df = df[df.kfold != fold].reset_index(drop=True)
    #valid_df = df[df.kfold == fold].reset_index(drop=True)
    train_df_fold = df.loc[fold_indices!=fold].reset_index(drop=True)
    valid_df_fold = df.loc[fold_indices==fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    #train_dataset = make_dataset(train_df, tokenizer, DEVICE)
    #valid_dataset = make_dataset(valid_df, tokenizer, DEVICE)
    train_dataset = MyDataset(data=train_df_fold, tokenizer=tokenizer, max_token_len=MAX_TOKEN_LEN)
    valid_dataset = MyDataset(data=valid_df_fold, tokenizer=tokenizer, max_token_len=MAX_TOKEN_LEN)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(n_classes=NUM_CLASSES, d_hidden_linear=D_HIDDEN_LINEAR,
                       dropout_rate=DROPOUT_RATE, pooling_type=POOLING_TYPE,
                       )
    model = model.to(DEVICE)

    #criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=EPOCHS * len(train_dataloader),
        num_warmup_steps=50
    )


    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_fn(train_dataloader, model, optimizer, scheduler, epoch)
        valid_loss, valid_acc, valid_f1 = eval_fn(valid_dataloader, model, epoch)
        print(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ", end="")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        plot_training(train_losses, train_accs, train_f1s,
                      valid_losses, valid_accs, valid_f1s,
                      epoch, fold)
        
        best_loss = valid_loss if valid_loss < best_loss else best_loss
        best_acc = valid_acc if valid_acc > best_acc else best_acc
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            print("model saving!", end="")
            torch.save(model.state_dict(), MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth")

        print("\n")

    return best_f1


def main():
    seed_everything(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))

    train_df = pd.read_csv(TRAIN_FILE)
    train_df["jobflag"] = train_df["jobflag"] - 1
    train_df = preprocess(train_df)
    test_df = pd.read_csv(TEST_FILE)
    test_df["jobflag"] = 0
    test_df = preprocess(test_df)

    trn_fold = range(NUM_SPLITS)
    if FOLD_TYPE == 'kf':
        fold_indices = get_kfold(train=train_df, n_splits=NUM_SPLITS, seed=SEED)
    if FOLD_TYPE == 'skf':
        fold_indices = get_stratifiedkfold(train=train_df, target_col='jobflag', n_splits=NUM_SPLITS, seed=SEED)

    # training
    f1_scores = []
    for fold in trn_fold:
        print(f"fold {fold}", "="*80)
        f1 = trainer(fold, fold_indices, train_df)
        f1_scores.append(f1)
        print(f"<fold={fold}> best score: {f1}\n")
    
    cv = sum(f1_scores) / len(f1_scores)
    print(f"CV: {cv}")

    lines = ""
    for i, f1 in enumerate(f1_scores):
        line = f"fold={i}: {f1}\n"
        lines += line
    lines += f"CV    : {cv}"
    with open(f"./result/{MODEL_NAME}_result.txt", mode='w') as f:
        f.write(lines)

    # inference
    models = []
    for fold in trn_fold:
        model = Classifier(n_classes=NUM_CLASSES, d_hidden_linear=D_HIDDEN_LINEAR,
                           dropout_rate=DROPOUT_RATE, pooling_type=POOLING_TYPE)
        model.load_state_dict(torch.load(MODELS_DIR + f"best_{MODEL_NAME}_{fold}.pth"))
        model.to(DEVICE)
        model.eval()
        models.append(model)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = MyDataset(data=test_df, tokenizer=tokenizer, max_token_len=MAX_TOKEN_LEN)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        progress = tqdm(test_dataloader, total=len(test_dataloader))
        final_output = []

        for batch in progress:
            progress.set_description("<Test>")
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask=batch["attention_mask"].to(DEVICE)
            #labels = batch["labels"].to(DEVICE)

            outputs = []
            for model in models:
                preds, _ = model.forward(input_ids=input_ids,
                                     attention_mask=attention_mask)
                outputs.append(preds)

            outputs = sum(outputs) / len(outputs)
            outputs = torch.softmax(outputs, dim=1).cpu().detach().tolist()
            outputs = np.argmax(outputs, axis=1)

            final_output.extend(outputs)
    
    submit = pd.read_csv(os.path.join(DATA_PATH, "submit_sample.csv"), names=["id", "jobflag"])
    submit["jobflag"] = final_output
    submit["jobflag"] = submit["jobflag"] + 1
    try:
        submit.to_csv("./output/submission_" + EXP_NUM + "_cv{}.csv".format(str(cv).replace(".", "")[:10]), index=False, header=False)
    except NameError:
        submit.to_csv("./output/submission.csv", index=False, header=False)


if __name__ == '__main__':
    main()