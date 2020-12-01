import os
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
            self, model, train_loader, val_loader,
            logdir, optimizer, scheduler, criterion,
            metrics, accumulation, activation
    ):
        '''
            model: pytorch model
            train_loader: pytorch train loader
            val_loader: pytorch val loader
            logdir: directory to store logs
            optimizer: pytorch optimizer
            scheduler: pytorch scheduler
            criterion: pytorch loss, takes loss(preds, true)
            metrics: dict of pytorch metrics, takes metric(preds, true)
            accumulation: dict of {k: v} where k - accumulation steps, v - proba of steps
            activation: function to apply after forward to evaluate
        '''
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logdir = logdir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = metrics
        self.accumulation = accumulation
        self.activation = activation

        try:
            os.stat(self.logdir)
        except:
            os.mkdir(self.logdir)
            os.mkdir(self.logdir + '/checkpoints')

    def loger(self, epoch, train_loss, val_loss, metrics, verbose=False):
        out_str = f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, ' \
                  + ', '.join([k + ': ' + str(v) for k, v in metrics.items()])
        if verbose:
            print(out_str)
        with open(self.logdir + '/metrics.txt', 'a+') as f:
            f.write(out_str + '\n')

    def train(self, n_epochs=1, verbose=False):
        model = self.model.cuda()
        train_loader = self.train_loader
        val_loader = self.val_loader
        logdir = self.logdir
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion
        metrics = self.metrics
        accumulation = self.accumulation
        accumulation_steps = []
        accumulation_p = []
        for k, v in accumulation.items():
            accumulation_steps.append(k)
            accumulation_p.append(v)
        activation = self.activation

        for epoch in tqdm(range(n_epochs)):
            model = model.train()
            model.zero_grad()
            optimizer.zero_grad()

            train_epoch_loss = []
            accumulation_step = np.random.choice(accumulation_steps, p=accumulation_p)
            for i, (x, y) in enumerate(tqdm(train_loader)):
                x = x.cuda()
                y = y.cuda()
                y_preds = model(x)
                loss = criterion(y_preds, y)
                train_epoch_loss.append(loss.item())
                loss.backward()
                if (i + 1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            train_epoch_loss = np.mean(train_epoch_loss)

            val_epoch_loss = []
            val_preds = []
            val_true = []
            with torch.no_grad():
                model = model.eval()
                torch.save(model.state_dict(), logdir + '/checkpoints/' + f'model_epoch_{epoch}.pth')
                for x, y in tqdm(val_loader):
                    x = x.cuda()
                    y = y.cuda()
                    y_preds = model(x)
                    loss = criterion(y_preds, y)
                    val_epoch_loss.append(loss.item())
                    val_preds.append(y_preds.detach().cpu())
                    val_true.append(y.detach().cpu())
            val_epoch_loss = np.mean(val_epoch_loss)
            val_preds = activation(torch.cat(val_preds))
            val_true = torch.cat(val_true)
            scheduler.step(val_epoch_loss)
            val_metrics = {name: metric(val_preds, val_true) for name, metric in metrics.items()}
            self.loger(epoch, train_epoch_loss, val_epoch_loss, val_metrics, verbose)