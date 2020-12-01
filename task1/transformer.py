import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from utils.pytorch_trainer import Trainer


class Model(nn.Module):
    def __init__(self, embed_size):
        super(Model, self).__init__()
        self.hidden_dim=128
        self.embed_size = embed_size
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=8, dim_feedforward=self.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        out = self.embedding(x)
        out = self.transformer(out)
        out = torch.mean(out, dim=2).squeeze()
        out = self.dropout(out)
        out = self.linear(out)
        return out
    
    
def fit_predict(data, max_seq_len):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    transformer_model = Model(X_train.shape[1] + max_seq_len) ## add pos embedding
    
    X_seq = []
    for ind in range(X_train.shape[1]):
        x_curr = X_train.iloc[max(0, ind-max_seq_len): ind + 1].values
        if x_curr.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - x_curr.shape[0], X_train.shape[1]))
            x_curr = np.vstack([padding, x_curr])
        pos_embedding = np.eye(max_seq_len) # pos embedding
        x_curr = np.hstack([x_curr, pos_embedding])
        
    X_seq = np.array(X_seq)
    
    X_seq_test = []
    for ind in range(X_test.shape[1]):
        x_curr = X_test.iloc[max(0, ind-max_seq_len): ind + 1].values
        if x_curr.shape[0] < max_seq_len:
            padding = X_train.iloc[-(max_seq_len - x_curr.shape[0])].values
            x_curr = np.vstack([padding, x_curr])
        pos_embedding = np.eye(max_seq_len) # pos embedding
        x_curr = np.hstack([x_curr, pos_embedding])
    X_seq_test = np.array(X_seq)


    batch_size=512
    train = TensorDataset(torch.tensor(X_seq), torch.tensor(y_train))
    test = TensorDataset(torch.tensor(X_seq_test), torch.tensor(y_test))
    train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=6)
    val_dataloader = DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=6)
    
    logdir = 'task1_transformer_baseline'
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = Adam(lr=1e-3, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, min_lr=1e-6, patience=3, verbose=True)
    metrics = {}
    activation = torch.sigmoid
    trainer = Trainer(
        model=transformer_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        logdir=logdir,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        metrics=metrics,
        accumulation={1: 1.0},
        activation=activation
    )
    val_preds = trainer.train(n_epochs=50, verbose=True)
    return val_preds
