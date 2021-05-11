import torch
from torchmetrics.functional import accuracy as metric_acc
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class pl_wrapper(pl.LightningModule):
    def __init__(self, batch_size = 512, nworkers = 8):
        super().__init__()
        self.batch_size = batch_size
        self.nworkers = nworkers
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = metric_acc(y, preds)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        _, preds = torch.max(out, 1)
        acc = metric_acc(y, preds)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        acc = metric_acc(y, preds)

        self.log('test_acc', acc, prog_bar = True, on_step=False, on_epoch = True, logger = True)

        return acc

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.nworkers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size = self.batch_size, shuffle=False, num_workers=self.nworkers)

if __name__ == '__main__':
    class test_model(pl_wrapper):
        def __init__(self, batch_size = 256, nworkers = 2):
            super().__init__(batch_size, nworkers)
            self.l1 = torch.nn.Linear(100, 10)
        
        def forward(x):
            x = self.l1(x)

            return x
    
    model = test_model()
    print(model.parameters)
    print(model.batch_size, model.nworkers)