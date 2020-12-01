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
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(2*self.hidden_dim, 1)
    
    def forward(self, x):
        out = self.embedding(x)
        out, hidden = self.lstm(out)
        out = torch.cat(hidden, dim=2).squeeze()
        out = self.dropout(out)
        out = self.linear(out)
        return out
    
    
def fit_predict(data, max_seq_len):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    lstm_model = Model(X_train.shape[1])
    
    X_seq = []
    for ind in range(X_train.shape[1]):
        x_curr = X_train.iloc[max(0, ind-max_seq_len): ind + 1].values
        if x_curr.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - x_curr.shape[0], X_train.shape[1]))
            x_curr = np.vstack([padding, x_curr])
    X_seq = np.array(X_seq)
    
    X_seq_test = []
    for ind in range(X_test.shape[1]):
        x_curr = X_test.iloc[max(0, ind-max_seq_len): ind + 1].values
        if x_curr.shape[0] < max_seq_len:
            padding = X_train.iloc[-(max_seq_len - x_curr.shape[0])].values
            x_curr = np.vstack([padding, x_curr])
    X_seq_test = np.array(X_seq)


    batch_size=512
    train = TensorDataset(torch.tensor(X_seq), torch.tensor(y_train))
    test = TensorDataset(torch.tensor(X_seq_test), torch.tensor(y_test))
    train_dataloader = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=6)
    val_dataloader = DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=6)
    
    logdir = 'task2_lstm_baseline'
    criterion = torch.nn.functional.mse_loss
    optimizer = Adam(lr=1e-3, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, min_lr=1e-6, patience=3, verbose=True)
    metrics = {}
    activation = lambda x: x
    trainer = Trainer(
        model=lstm_model,
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
