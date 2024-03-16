from typing import Callable, Union, Optional

import lightning as L
from lightning.pytorch import loggers
from torchmetrics.aggregation import CatMetric

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def predicted_vs_actual(predicted: torch.Tensor|np.ndarray|list,
                        actual: torch.Tensor|np.ndarray|list) -> matplotlib.figure.Figure:
    """
    Generate a scatter plot depicting the relationship between predicted and actual values
    for a response variable.
    
    Args:
        predicted (torch.Tensor | np.ndarray | list): vector of predicted values.
        actual (torch.Tensor | np.ndarray | list): vector of actual values.
        
    Returns:
        matplotlib.figure.Figure: A Matplotlib Figure object representing the scatter plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(predicted), np.array(actual))
    ax.axline((1.5, 1.5), (1.51, 1.51), color='r')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ẑ')
    ax.set_ylabel('Z')
    return fig


#TO-DO: implementar uma classe de classificacao.
# testar lr scheduler
class RegressionModule(L.LightningModule):
    """
    LightningModule class using a custom Pytorch Module for regression task.
    
    Args:
            pytorch_module (nn.Module): Pytorch Module.
            weight_decay (float, optional): _description_. Defaults to 0.0.
            lr (float, optional): _description_. Defaults to None.
            lr_patience (int, optional): _description_. Defaults to 1.
            lr_factor (float, optional): _description_. Defaults to 1.0.
            tb_dir (str, optional): _description_. Defaults to ''.
    
    """
    def __init__(self, *args, pytorch_module: nn.Module,
                 loss_func: Union[Callable[..., None], str], loss_func_args: dict={},
                 optimizer: Union[Callable[..., None], str], optimizer_args: dict={},
                 lr_scheduler: Optional[Union[Callable[..., None], str]]=None,
                 lr_scheduler_args: dict={},
                 tb_dir='',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pytorch_module = pytorch_module
        
        if isinstance(loss_func, str):
            self.loss_func = eval("nn." + loss_func + "(**loss_func_args)")
        else:
            self.loss_func = loss_func(**loss_func_args)
            
        if isinstance(optimizer, str):
            self.optimizer = eval("torch.optim."
                                  + optimizer
                                  + "(self.parameters(), **optimizer_args)")
        else:
            self.optimizer = optimizer(self.parameters(), **optimizer_args)
            
        if lr_scheduler is None:
            self.lr_scheduler = None
        elif isinstance(lr_scheduler, str):
            self.lr_scheduler = eval("torch.optim.lr_scheduler."
                                     + lr_scheduler
                                     + "(self.optimizer, **lr_scheduler_args)")
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_args)

        self.tb = loggers.TensorBoardLogger(save_dir=tb_dir)

    def forward(self, x):
        return self.pytorch_module(x)

    def on_train_epoch_start(self):
        self.current_epoch_train_targets = CatMetric()
        self.current_epoch_train_outputs = CatMetric()

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)

        self.log("train_loss", loss, logger=True)

        self.current_epoch_train_targets.update(torch.flatten(target))
        self.current_epoch_train_outputs.update(torch.flatten(output))

        return loss

    def on_train_epoch_end(self):
        epoch_loss = self.loss_func(self.current_epoch_train_targets.compute(),
                                    self.current_epoch_train_outputs.compute())

        self.tb.experiment.add_scalars("losses", {"training_loss": epoch_loss},
                                       global_step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.current_epoch_validation_targets = CatMetric()
        self.current_epoch_validation_outputs = CatMetric()

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)

        self.log("val_loss", loss, logger=True)

        self.current_epoch_validation_targets.update(torch.flatten(target))
        self.current_epoch_validation_outputs.update(torch.flatten(output))

        return loss

    def on_validation_epoch_end(self):
        epoch_loss = self.loss_func(self.current_epoch_validation_outputs.compute(),
                                    self.current_epoch_validation_targets.compute())

        self.tb.experiment.add_scalars("losses", {"validation_loss": epoch_loss},
                                       global_step=self.current_epoch)
        
    def on_validation_end(self):
        actual = self.current_epoch_validation_targets.compute() #last epoch
        predicted = self.current_epoch_validation_outputs.compute() #last epoch

        desv_var = np.var(np.array(actual) - np.array(predicted))

        self.tb.experiment.add_scalar("desvio_var (validation)", desv_var)

        self.tb.experiment.add_figure('predicted vs. actual (validation)',
                                      predicted_vs_actual(predicted, actual))

    def on_test_epoch_start(self):
        self.current_epoch_test_targets = CatMetric()
        self.current_epoch_test_outputs = CatMetric()

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)

        self.current_epoch_test_targets.update(torch.flatten(target))
        self.current_epoch_test_outputs.update(torch.flatten(output))

        return loss

    def on_test_epoch_end(self):
        actual = self.current_epoch_test_targets.compute()
        predicted = self.current_epoch_test_outputs.compute()

        test_loss = self.loss_func(actual, predicted)
        desv_var = np.var(np.array(actual) - np.array(predicted))

        self.tb.experiment.add_scalar("test_loss", test_loss)
        self.tb.experiment.add_scalar("desvio_var (test)", desv_var)

        self.tb.experiment.add_figure('predicted vs. actual (test)',
                                      predicted_vs_actual(predicted, actual))

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        